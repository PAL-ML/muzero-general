
# when ray terminates the worker, how are the spawned processes handled? are they terminated by default?
# answer: if we use muzero terminate_workers, the processes will exit on their own

# TODO: add imports lol

import ray
import torch # can theoretically remove
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import time

import replay_buffer
import self_play
import trainer

import models


# arguments for xmp.spawn 
N_PROC = 1
START_METHOD = "fork"


# TODO: refactor these names to something more logical
@ray.remote(resources={"tpu": 1})
def runSelfPlayWrapped(checkpoint, game, config, replay_buffer_worker, shared_storage_worker):
	# TODO: logging loop!

	def map_fn(index):
		print("selfplay instantiation begins")
		self_play_worker = self_play.SelfPlay(checkpoint, game, config, config.seed)
		print("selfplay instantiation begins")

		# when we have multiple self-play workers, we'll want this to happen only when all of them
		# are ready. we can achieve that using rendezvous and taking advantage of spawn's blocking 
		shared_storage_worker.set_info.remote("trainer_can_start", True)
		
		print("selfplay continuous beginning")
		self_play_worker.continuous_self_play(shared_storage_worker, replay_buffer_worker)

	# map_fn(None)

	xmp.spawn(
		map_fn,
		args=(),
		nprocs=N_PROC,
		start_method=START_METHOD
		)

@ray.remote(resources={"tpu": 1})
def runTrainerWrapper(checkpoint, config, replay_buffer_worker, shared_storage_worker):
	def map_fn(index):
		c = 0

		while not ray.get(shared_storage_worker.get_info.remote("trainer_can_start")) and c < 120:
			# print(f"fetch false, sleeping ({c})")
			time.sleep(10)
			c += 1

		if not ray.get(shared_storage_worker.get_info.remote("trainer_can_start")):
			raise Exception("Timeout while waiting for hook to yeet rip")
    
		print("trainer instantiation begins")
		training_worker = trainer.Trainer(checkpoint, config)
		print("trainer instantiation done! starting weight updates")
		training_worker.continuous_update_weights(
			replay_buffer_worker, shared_storage_worker
		)

	# map_fn(None)

	xmp.spawn(
		map_fn,
		args=(),
		nprocs=1, 
		start_method=START_METHOD
		)

# TODO: migrate this to a function and generally get it working
@ray.remote
class ReanalyseWrapper():
	@staticmethod
	def _map_fn(index, config, checkpoint, shared_storage_worker, replay_buffer_worker):
		reanalyse_worker = replay_buffer.Reanalyse(self.checkpoint, self.config)

		# i think this will make it wait for every worker to instantiate before it starts runs
		# xm.rendezvous('init')

		reanalyse_worker.reanalyse(
			replay_buffer_worker, shared_storage_worker
		)

	def run(self, config, checkpoint, shared_storage_worker, replay_buffer_worker):
		xmp.spawn(self._map_fn, args=(config, checkpoint, shared_storage_worker, replay_buffer_worker), nprocs=N_PROC, start_method=START_METHOD)

"""
inside simple_map_fn, we want:
- instantiation of the model
- copy the model to device
- run the actual training
"""

