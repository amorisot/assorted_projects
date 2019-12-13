import modified_data_providers as data_providers


# I used to flatten the targets, so be sure I don't do that
def test_mtl_smallnorb_target_shape():
	BATCH_SIZE = 50
	NUM_DEVICES = 1

	train_data = data_providers.MTLSmallNORBDataProvider('train', batch_size=BATCH_SIZE, flatten=True, num_devices=NUM_DEVICES)
	input_batch, target_batch = train_data.next()

	assert target_batch.shape == (NUM_DEVICES, BATCH_SIZE, 2)


def test_multi_gpu_smallnorb_target_shape():
	BATCH_SIZE = 50
	NUM_DEVICES = 8

	train_data = data_providers.SmallNORBDataProvider('train', batch_size=BATCH_SIZE, flatten=True, num_devices=NUM_DEVICES)
	input_batch, target_batch = train_data.next()

	assert target_batch.shape == (NUM_DEVICES, BATCH_SIZE)
