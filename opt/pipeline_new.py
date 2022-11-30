from pipeline_utils import *
import argparse

# Init Config
parser = argparse.ArgumentParser()
args, gpu_id, device = get_svox2_config(parser)
cfg_convonet = get_convonet_config(args)
pipeline = Pipeline(args, cfg_convonet)

# 在这种模式下，最好先给定场景的BBox
# 当前会Crop by z
# Get OccGrid by convonet
grid_reso = cfg_convonet['model']['grid_resolution']
print("ConvONet gird_reso = ", grid_reso)
pipeline.init_convonet(cfg_convonet)
grid_bbox, n_crop_axis = pipeline.get_grid_bbox()
torch.cuda.empty_cache()

# @TODO:要根据RGBD，分批次进行优化！！！
# Plenoxel Init
pipeline.svox2_init(grid_bbox, data_split="train") # loding data and get resolution 
pipeline.svox2_init_grid(128, torch.ones(list(n_crop_axis*128)).int().to(device) * -1)
pipeline.svox2_load_dataset(grid_bbox, split="test")
# occ_grid_links, _ = pipeline.get_convonet_occ(crop_reso=128, occ_by_net=True)
# pipeline.svox2_update_grid(occ_grid_links)
torch.cuda.empty_cache()
n_images = pipeline.svox2["dset"].n_images
image_crop = 50
overlap = 5
# pipeline.svox2_train(2)

for i in range(0, n_images, image_crop):
    i_end = min(i+image_crop+overlap, n_images)
    i_start = max(0, i - overlap)
    print(f"Train image range = [{i_start}, {i_end}]")
    points = pipeline.get_pointcloud_from_svox2_dset(i, i_end)
    # np.savetxt(os.path.join(args.train_dir, f"points_{i}.txt"), points[0].cpu().numpy())
    # occ_grid_links, _ = pipeline.get_convonet_occ(input = points, crop_reso=128, occ_by_net=False)
    occ_grid_links, _ = pipeline.get_convonet_occ(input = points, crop_reso=128, occ_by_net=False)
    # torch.save(occ_grid_links.cpu(),os.path.join(args.train_dir, f"occ_links_{i}.ckpt") )
    pipeline.svox2_update_grid(occ_grid_links)
    cam_list = pipeline.links_get_cam()
    # cam_list = torch.arange(i, i_end)
    print("cam_list num = ", cam_list.shape[0])
    print("cam_list = ", cam_list)
    pipeline.svox2_train(1, cam_list)
    # torch.cuda.empty_cache()

# Loading test data and eval
pipeline.svox2_eval()
pipeline.svox2_save()
print("Done")