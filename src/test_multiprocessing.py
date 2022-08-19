import numpy as np
import ray

ray.init(num_cpus = 4)

@ray.remote
def test(img, slice_idx):
    print(f'current slice is {slice_idx}')
    return sum(img[slice_idx,:,:])


if __name__ == "__main__":
    img = np.random.randint(0,5,(10, 3, 3))
    img_shared = ray.put(img)

    result_ids = ray.get([test.remote(img_shared, x) for x in range(10)])
    ray.shutdown()
