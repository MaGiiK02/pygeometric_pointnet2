
def limitedExponentialDecayLR(batch_size, lr_decay, decay_step):
	return lambda it: max(
		lr_decay ** (int(it * batch_size / decay_step)),
		1e-5
	)