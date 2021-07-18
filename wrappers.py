
# when ray terminates the worker, how are the spawned processes handled? are they terminated by default?
# answer: if we use muzero terminate_workers, the processes will exit on their own

# TODO: add imports lol

import ray
import torch # can theoretically remove
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from abc import ABC

import replay_buffer
import self_play
import trainer

import models


# arguments for xmp.spawn 
N_PROC = 1
START_METHOD = "fork"


@ray.remote
class SelfPlayWrapper():
	# does this actually need an init? no right?

	@staticmethod
	def _map_fn(index, num_gpus_per_worker, config, checkpoint, game, shared_storage_worker, replay_buffer_worker, test_mode):
		# so we it should only spawn one here right? since spawn is taking care of creating more
		# than one process
		self_play_worker = self_play.SelfPlay(checkpoint, game, config, config.seed)
		print("SELFPLAY IS WORKING BABY")
		
		# i think this will make it wait for every worker to instantiate before it starts runs
		# xm.rendezvous('init')

		# self.model.to(self.device)
		# self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

		# if test_mode:
		# 	self_play_worker.continuous_self_play(shared_storage_worker, None, True)
		# else:
		# 	self_play_worker.continuous_self_play(shared_storage_worker, replay_buffer_worker)

	def run(self, num_gpus_per_worker, config, checkpoint, game, shared_storage_worker, replay_buffer_worker, test_mode=False):
                if test_mode: # True
	                self._map_fn(0, num_gpus_per_worker, config, checkpoint, game, shared_storage_worker, replay_buffer_worker, test_mode)
                else:
                        xmp.spawn(
			    self._map_fn, 
		    	    args=(num_gpus_per_worker, config, checkpoint, game, shared_storage_worker, replay_buffer_worker, test_mode), 
			    nprocs= 1 if test_mode else  N_PROC, # if it's for the logging loop, we only want one
			    start_method=START_METHOD
			    )

@ray.remote
class TrainerWrapper():
	@staticmethod
	def _map_fn(index, config, checkpoint, shared_storage_worker, replay_buffer_worker):
		training_worker = trainer.Trainer(checkpoint, config)
		print("TRAINING IS WORKING BABY")

		# i think this will make it wait for every worker to instantiate before it starts runs
		# xm.rendezvous('init')

		# training_worker.continuous_update_weights(
		# 	replay_buffer_worker, shared_storage_worker
		# )

	def run(self, config, checkpoint, shared_storage_worker, replay_buffer_worker):
                # self._map_fn(0, config, checkpoint, shared_storage_worker, replay_buffer_worker)
	        xmp.spawn(self._map_fn, args=(config, checkpoint, shared_storage_worker, replay_buffer_worker), nprocs=N_PROC, start_method=START_METHOD)

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


