import datetime
import os

import matplotlib.pyplot as plt
import cv2
import numpy as np
import geopandas as gpd
from eolearn.core import (
    EOTask, 
    EOPatch,
    EOWorkflow,
    FeatureType,
    MapFeatureTask,
    RemoveFeatureTask,
    linearly_connect_tasks,
    EOExecutor,
)
from eolearn.features import SimpleFilterTask
from eolearn.io import SentinelHubInputTask
from eolearn.features.utils import spatially_resize_image as resize_images
from sentinelhub import (
    BBox,
    DataCollection,
    SHConfig,
    get_utm_crs,
    wgs84_to_utm,
)
from sentinelhub.exceptions import SHDeprecationWarning
from tqdm.auto import tqdm

from phisat2_constants import (
    S2_BANDS,
    S2_RESOLUTION,
    BBOX_SIZE,
    PHISAT2_RESOLUTION,
    ProcessingLevels,
    WORLD_GDF,
)
from phisat2_utils import (
    AddPANBandTask,
    AddMetadataTask,
    CalculateRadianceTask,
    CalculateReflectanceTask,
    SCLCloudTask,
    BandMisalignmentTask,
    PhisatCalculationTask,
    AlternativePhisatCalculationTask,
    CropTask,
    GriddingTask,
    ExportGridToTiff,
    get_extent,
)

PROCESSING_LEVEL = ProcessingLevels.L1C

# sh_config = SHConfig()
# sh_config.sh_client_id = "<your Sentinel Hub OAuth client ID>"
# sh_config.sh_client_secret = "<your Sentinel Hub OAuth client secret>"


# def get_utm_bbox(lat_centre: float, lon_centre: float):
#     """Returns a bounding box of size corresponding to the swath width of Φ-sat-2 given the centroid of the area-of-interest in WGS84"""

#     east, north = wgs84_to_utm(lon_centre, lat_centre)

#     east, north = 10 * int(east / 10), 10 * int(north / 10)
#     crs = get_utm_crs(lon_centre, lat_centre)

#     return BBox(
#         bbox=(
#             (east - BBOX_SIZE // 2, north - BBOX_SIZE // 2),
#             (east + BBOX_SIZE // 2, north + BBOX_SIZE // 2),
#         ),
#         crs=crs,
#     )


# lat_centre, lon_centre = 42.348, 13.397  # l'Aquila
# bbox = get_utm_bbox(lat_centre, lon_centre)

# time_interval = ("2021-09-01", "2021-09-10")
# maxcc = 0.7  # The maximum allowed cloud coverage in percent
# aux_request_args = {"processing": {"upsampling": "BICUBIC"}}


# scl_download_task = SentinelHubInputTask(
#     data_collection=DataCollection.SENTINEL2_L2A,
#     resolution=S2_RESOLUTION,
#     additional_data=[(FeatureType.MASK, "SCL")],
#     maxcc=maxcc,
#     aux_request_args=aux_request_args,
#     config=sh_config,
#     cache_folder="./temp_data/",
#     time_difference=datetime.timedelta(minutes=180),
# )
# scl_cloud_task = SCLCloudTask(scl_feature=(FeatureType.MASK, "SCL"))

# eop = scl_download_task(bbox=bbox, time_interval=time_interval)
# eop = scl_cloud_task(eopatch=eop)


# additional_bands = [(FeatureType.DATA, name) for name in ["sunZenithAngles"]]
# masks = [(FeatureType.MASK, "dataMask")]


# input_task = SentinelHubInputTask(
#     data_collection=DataCollection.SENTINEL2_L1C,
#     resolution=S2_RESOLUTION,
#     bands_feature=(FeatureType.DATA, "BANDS"),
#     additional_data=masks + additional_bands,
#     bands=S2_BANDS,  # note the order of these bands, where B08 follows B03
#     aux_request_args=aux_request_args,
#     config=sh_config,
#     cache_folder="./temp_data/",
#     time_difference=datetime.timedelta(minutes=180),
# )

# eop = input_task(eopatch=eop)

# add_meta_task = AddMetadataTask(config=sh_config)
# eop = add_meta_task(eop)

# eop = EOPatch.load('phisat-2/data/demo_input_25_singular')

# radiance_task = CalculateRadianceTask(
#     (FeatureType.DATA, "BANDS"), (FeatureType.DATA, "BANDS-RAD")
# )
# eop = radiance_task(eop)

# add_pan_task = AddPANBandTask(
#     (FeatureType.DATA, "BANDS-RAD"), (FeatureType.DATA, "BANDS-RAD-PAN")
# )
# eop = add_pan_task(eop)

# features_to_resize = {
#     FeatureType.DATA: ["BANDS-RAD-PAN", "sunZenithAngles"],
#     FeatureType.MASK: [
#         "SCL_CLOUD",
#         "SCL_CIRRUS",
#         "SCL_CLOUD_SHADOW",
#         "dataMask",
#     ],
# }
# NEW_SIZE = (int(BBOX_SIZE / PHISAT2_RESOLUTION), int(BBOX_SIZE / PHISAT2_RESOLUTION))

# casting_task = MapFeatureTask(
#     (FeatureType.MASK, "dataMask"), (FeatureType.MASK, "dataMask"), np.uint8
# )
# eop = casting_task(eop)

# resize_task_list = []
# for feature_type in tqdm(features_to_resize.keys()):
#     for feature in tqdm(features_to_resize[feature_type]):
#         resize_task_list.append(
#             MapFeatureTask(
#                 (feature_type, feature),
#                 (feature_type, f"{feature}_RES"),
#                 resize_images,
#                 new_size=NEW_SIZE,
#                 resize_method="nearest",
#             )
#         )
#         eop = resize_task_list[-1](eop)


# remove_feature_task1 = RemoveFeatureTask(
#     [
#         (FeatureType.DATA, "BANDS"),
#         (FeatureType.DATA, "BANDS-RAD"),
#         (FeatureType.DATA, "BANDS-RAD-PAN"),
#         (FeatureType.DATA, "sunZenithAngles"),
#         (FeatureType.MASK, "SCL_CLOUD"),
#         (FeatureType.MASK, "SCL_CLOUD_SHADOW"),
#         (FeatureType.MASK, "SCL_CIRRUS"),
#         (FeatureType.MASK, "dataMask"),
#     ]
# )
# eop = remove_feature_task1(eop)
# eop.save('phisat-2/data/band_shift_input')

eop = EOPatch.load('phisat-2/data/band_shift_input')

band_shift_task = BandMisalignmentTask(
    (FeatureType.DATA, "BANDS-RAD-PAN_RES"),
    (FeatureType.DATA, "BANDS-RAD-PAN_SHIFTED"),
    PROCESSING_LEVEL,
    std_sea=6,
    interpolation_method=cv2.INTER_NEAREST,
)
eop = band_shift_task(eop)

for shifts in eop.meta_info["Shifts"].values():
    print(np.round(shifts))
print(1/0)

# plot the shift in bands
ts_idx = 1
num_bands = 8

for b_idx in range(0, num_bands):
    _, ax = plt.subplots(ncols=3, nrows=1, figsize=(15, 15), sharex=True, sharey=True)
    ax[0].imshow(
        eop.data["BANDS-RAD-PAN_RES"][ts_idx][-100:, :100, b_idx] / 100, vmin=0, vmax=1
    )
    ax[0].set_title(f"Original band - index {b_idx}")

    ax[1].imshow(
        eop.data["BANDS-RAD-PAN_SHIFTED"][ts_idx][-100:, :100, b_idx] / 100,
        vmin=0,
        vmax=1,
    )
    ax[1].set_title(f"Shifted band- index {b_idx}")

    diff = (
        eop.data["BANDS-RAD-PAN_RES"][ts_idx][-100:, :100, b_idx]
        - eop.data["BANDS-RAD-PAN_SHIFTED"][ts_idx][-100:, :100, b_idx]
    )
    ax[2].imshow(diff)
    ax[2].set_title("Difference")


# In[44]:


remove_feature_task2 = RemoveFeatureTask([(FeatureType.DATA, "BANDS-RAD-PAN_RES")])


# In[45]:


eop = remove_feature_task2(eop)
eop


# Crop the image to contain only valid data after band misalignment

# In[46]:


crop_task = CropTask(
    features_to_crop=[
        (FeatureType.DATA, "BANDS-RAD-PAN_SHIFTED"),
        (FeatureType.DATA, "sunZenithAngles_RES"),
        (FeatureType.MASK, "SCL_CLOUD_RES"),
        (FeatureType.MASK, "SCL_CLOUD_SHADOW_RES"),
        (FeatureType.MASK, "SCL_CIRRUS_RES"),
        (FeatureType.MASK, "dataMask_RES"),
    ]
)


# In[47]:


eop = crop_task(eop)

eop


# <a name="snr"></a>
# ## 7. Noise Calculation
# 
# This step simulates noise according to the signal-toNoise (SNR) specifications of the optical system.

# A compiled executable will perform the SNR and PSF calculation. 
# 
# Executables for the following Operating Systems (OSs) have been compiled:
# 
#  * Unix, `phisat2_unix.bin`
#  * Windows, `phisat2_win.bin`
#  * MacOS, `phisat2_osx-arm64.bin` for ARM chips, `phisat2_osx-x86_64.bin` for Intel chips
#  
# Download the suitable binary for your OS from this [link](https://cloud.sinergise.com/s/g88Ns32rXB3AT6i). 
# 
# Once downloaded, allow the operating system to run the file and make it executable (e.g. `chmod a+x phisat2_unix.bin`). Then set the path to the executable in the cell below.

# In[48]:


EXEC_PATH = "executables/phisat2_osx-arm64.bin"


# In[49]:


snr_task = PhisatCalculationTask(
    input_feature=(FeatureType.DATA, "BANDS-RAD-PAN_SHIFTED"),
    output_feature=(FeatureType.DATA, "L_out_RES"),
    executable=EXEC_PATH,
    calculation="SNR",
)


# In[51]:


eop = snr_task.execute(eop)

# check 
eop


# In[52]:


# plot differences due to noise
tidx = 1

_, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 15), sharex=True, sharey=True)

axes[0].imshow(eop.data["BANDS-RAD-PAN_SHIFTED"][t_idx][..., [2, 1, 0]] / 100.0)
axes[0].set_title(f"BEFORE - {timestamp}")
axes[1].imshow(eop.data["L_out_RES"][t_idx][..., [2, 1, 0]] / 100.0)
axes[1].set_title(f"AFTER - {timestamp}")
axes[2].imshow(
    np.abs(
        eop.data["L_out_RES"][t_idx][..., [2, 1, 0]]
        - eop.data["BANDS-RAD-PAN_SHIFTED"][t_idx][..., [2, 1, 0]]
    )
)
axes[2].set_title("RGB difference")
plt.tight_layout()


# <a name="psf"></a>
# ## 8. Simulate Point Spread Function (PSF) Filtering
# 
# Step that simulates loss of signal due to limited bandwidth of the Module Transfer Function (MTF), which is equivalent to simulate the Point Spread Function (PSF) for the entire system (imager and platform).
# 
# The same executable as for SNR is used. If you have not downloaded it already, do so for your OS from this [link](https://cloud.sinergise.com/s/g88Ns32rXB3AT6i). Once downloaded, allow the operating system to run the file and make it executable (e.g. `chmod a+x phisat2_unix.bin`). Then set the path to the executable in the cell below.

# In[53]:


psf_filter_task = PhisatCalculationTask(
    input_feature=(FeatureType.DATA, "L_out_RES"),
    output_feature=(FeatureType.DATA, "L_out_PSF"),
    executable=EXEC_PATH,
    calculation="PSF",
)


# In[54]:


eop = psf_filter_task(eop)

# check result:
eop


# In[55]:


# visualize part of EOPatch before/after applying PSF kernel

n_timestamps = len(eop.timestamp)
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 5))

axes[0].imshow(eop.data["L_out_RES"][0][2000:2200, 2000:2200, [2, 1, 0]] / 100)
axes[0].set_title(timestamp)
axes[1].imshow(eop.data["L_out_PSF"][0][2000:2200, 2000:2200, [2, 1, 0]] / 100)
axes[1].set_title("with PSF applied")
diff = eop.data["L_out_PSF"] - eop.data["L_out_RES"]
axes[2].hist(diff.ravel(), bins=40, log=True)
axes[2].set_title("difference, all bands and timestamps")

plt.tight_layout()


# In[56]:


eop


# In[61]:


# visualize part of EOPatch before/after applying PSF kernel

n_timestamps = len(eop.timestamp)
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 5))

axes[0].imshow(eop.data["L_out_RES"][0][2000:2200, 2000:2200, [2, 1, 0]] / 100)
axes[0].set_title(timestamp)
axes[1].imshow(eop.data["L_out_PSF"][0][2000:2200, 2000:2200, [2, 1, 0]] / 100)
axes[1].set_title("with PSF applied")
diff = eop.data["L_out_PSF"] - eop.data["L_out_RES"]
axes[2].hist(diff.ravel(), bins=40, log=True)
axes[2].set_title("difference, all bands and timestamps")

plt.tight_layout()


# <a name="alternative"></a>
# ## 9. Alternative Noise and PSF calculation 

# The `AlternativePhisatCalculationTask` allows users to input their own Signal To Noise ratios and their own Point Spread Function kernel values for addition of SNR and PSF.

# In[62]:


kernel_bands = ["B1", "B2", "B3", "B0", "B7", "B4", "B5", "B6"]
psf_kernel = { band: np.random.random(size=(7,7)) for band in kernel_bands }

snr_bands = ["B02", "B03", "B04", "PAN", "B08", "B05", "B06", "B07"]
snr_values = { band: np.random.randint(20,250) for band in snr_bands }

snr_psf_task = AlternativePhisatCalculationTask(
    input_feature=(FeatureType.DATA, "BANDS-RAD-PAN_SHIFTED"),
    snr_feature=(FeatureType.DATA, "L_out_RES"),
    psf_feature=(FeatureType.DATA, "L_out_PSF"),
    snr_values=snr_values,
    psf_kernel=psf_kernel,
    l_ref=100
)

# uncomment to run
# eop = snr_psf_task(eop)

# check result:
eop


# <a name="l1c"></a>
# ## 10. Compute L1C 
# 
# 
# If the specified processing level is L1C, reflectances are computed from radiances. Otherwise, radiances are provided as output for L1A and L1B.

# In[63]:


reflectance_task = CalculateReflectanceTask(
    input_feature=(FeatureType.DATA, "L_out_PSF"),
    output_feature=(FeatureType.DATA, "PHISAT2-BANDS"),
    processing_level=PROCESSING_LEVEL,
)


# In[64]:


eop = reflectance_task(eop)

eop


# In[65]:


# plot difference dub to radiance to refelectance conversion if applied
t_idx = 1
viz_factor = 3.5 if PROCESSING_LEVEL.value == ProcessingLevels.L1C.value else 0.01
_, axes = plt.subplots(ncols=2, nrows=1, figsize=(15, 15), sharex=True, sharey=True)

axes[0].imshow(eop.data["L_out_PSF"][t_idx][..., [2, 1, 0]] / 100.0)
axes[0].set_title(f"L_out_PSF - {timestamp}")
axes[1].imshow(eop.data["PHISAT2-BANDS"][t_idx][..., [2, 1, 0]] * viz_factor)
axes[1].set_title(f"PHISAT2-BANDS - {timestamp}")
plt.tight_layout()


# In[66]:


remove_feature_task3 = RemoveFeatureTask(
    [
        (FeatureType.DATA, "BANDS-RAD-PAN_SHIFTED"),
        (FeatureType.DATA, "L_out_PSF"),
        (FeatureType.DATA, "L_out_RES"),
    ]
)


# In[67]:


eop = remove_feature_task3(eop)


# <a name="gridding"></a>
# ## 11. Gridding 
# 
# On board of Φ-sat-2 the images covering the swath width will be split into smaller patches for computational processing. For creation of the training dataset, non-overlapping image patches of size 256x256 are created. When testing your algorithm, the image patches should possibly overlap to remove edge artefacts. 
# 
# Change the parameters in the cell below to modify the size of the image patches and their overlap.

# In[60]:


CELL_SIZE = 256
GRID_OVERLAP = 0.0


# The gridding task works for a single timestamp at a time. If you needed to grid multiple timeframes, you need to have a task for each timeframe to grid and export to Tiff.

# In[61]:


time_index = 0

FEATURE_NAMES = [
    (FeatureType.DATA, "PHISAT2-BANDS"),
    (FeatureType.MASK, "SCL_CIRRUS_RES"),
    (FeatureType.MASK, "SCL_CLOUD_RES"),
    (FeatureType.MASK, "SCL_CLOUD_SHADOW_RES"),
    (FeatureType.MASK, "dataMask_RES"),
]

gridding_tasks = []

for feature_type, feature_name in FEATURE_NAMES:
    gridding_tasks.append(
        GriddingTask(
            (feature_type, feature_name),
            (feature_type, f"{feature_name}-GRID_{time_index}"),
            (FeatureType.VECTOR_TIMELESS, f"{feature_name}-GRID_{time_index}"),
            size=CELL_SIZE,
            overlap=GRID_OVERLAP,
            resolution=PHISAT2_RESOLUTION,
            time_index=time_index,
        )
    )


# In[62]:


eop = gridding_tasks[0](eop)


# In[63]:


NSAMPLES = 9
_, axs = plt.subplots(
    figsize=(15, 15), ncols=3, nrows=NSAMPLES // 3, sharex=True, sharey=True
)
for nsample in range(NSAMPLES):
    axs[nsample % 3][nsample // 3].imshow(
        eop.data[f"PHISAT2-BANDS-GRID_{time_index}"][nsample][..., [2, 1, 0]]
        * viz_factor
    )
plt.tight_layout()


# In[64]:


# show resulting grid

fig, axs = plt.subplots(figsize=(15, 15))
axs.imshow(
    eop.data["PHISAT2-BANDS"][1][..., [2, 1, 0]] * viz_factor, extent=get_extent(eop)
)
eop.vector_timeless[f"PHISAT2-BANDS-GRID_{time_index}"].boundary.plot(ax=axs);


# <a name="export"></a>
# ## 12. Export to Tiff
#  
# Creates tiffs for a single timestamp. The simulated bands along with the cloud, cloud shadow, and cirrus mask are exported as tiff files. These files can be visualised and further analysed into any GIS software, such as QGis.
# 
# The format of the tiff filename is as follows:
# 
# ``{utm_easting_centroid}-{utm_northing_centroid}_{epsg_crs}_{feature}_{datetime}_{cell_number}``

# In[65]:


# create a folder for output tiffs
os.makedirs("tiff_folder", exist_ok=True)


# In[66]:


export_tiff_tasks = []

for feature_type, feature_name in FEATURE_NAMES:
    export_tiff_tasks.append(
        ExportGridToTiff(
            data_stack_feature=(feature_type, f"{feature_name}-GRID_{time_index}"),
            grid_feature=(
                FeatureType.VECTOR_TIMELESS,
                f"{feature_name}-GRID_{time_index}",
            ),
            out_folder="./tiff_folder",
            time_index=time_index,
        )
    )


# In[67]:


eop = export_tiff_tasks[0](eop)


# In[68]:


# check result:
os.listdir("tiff_folder")[:10]


# <a name="workflow"></a>
# ## 13. Create Workflow 
# 
# The tasks constituting the simulation workflow can be sequentially connected to create an EOWorkflow, which allows parallel execution over several AOIs. Read more about `eo-learn` workflows and parallel execution [here](https://eo-learn.readthedocs.io/en/latest/examples/core/CoreOverview.html#EONode-and-EOWorkflow). 

# In[69]:


nodes = linearly_connect_tasks(
    scl_download_task,
    scl_cloud_task,
    input_task,
    # filter_task,  # Uncomment this if you want to filter by 100% of valid data
    add_meta_task,
    radiance_task,
    add_pan_task,
    casting_task,
    *resize_task_list,
    remove_feature_task1,
    band_shift_task,
    remove_feature_task2,
    crop_task,
    snr_task,
    psf_filter_task,
    reflectance_task,
    remove_feature_task3,
    *gridding_tasks,
    *export_tiff_tasks,
)
workflow = EOWorkflow(nodes)


# In[ ]:


# This line repeats the simulation as done in the cells above
# results = workflow.execute({nodes[0]: {"bbox": bbox, "time_interval": time_interval}})


# <a name="core-dataset"></a>
# ## 14. Core Dataset
# 
# To get you started with your AI application at-the-edge, we simulated Φ-sat-2 imagery for around 490 location on Earth. The workflow as defined above was used to simulate L1C products for the locations and dates as specified in the provided `phisat2-locations.gpkg` file. The code below was used to generate such images. 
# 
# The locations and acquisition times seek to match the ones for the PRISMA images utilized in the IMAGIN-e track of the challenge. The information about the time difference from the PRISMA acquisition time is stored in the GeoDataFrame, as well as the `maxcc` of Sentinel-2 image and the overlap between the Φ-sat-2 bounding box and the geometry of the Sentinel-2 image. The dataframe also contains the name of the coresponding PRISMA image `prisma_name`, in case you wanted to cross-reference the datasets.
# 
# We suggest you to prepare a similar dataframe with dates and locations suitable for your application and use the code below to simulate Φ-sat-2 imagery. 
# 
# For the locations provided we suggest you download directly the provided tiff files, unless you want to tweak the simulation workflow. The name of the zip archive containing the image patches can be found in the `zipfile` column, so you can retrieve from storage only the simulated locations of interest. The zipfiles are available on cloudferro on the following public bucket 
# 
# `https://s3.waw2-1.cloudferro.com/swift/v1/AUTH_afccea586afd4ef3bb11fe37dd1ddfbf/OrbitalAI_Data/`, 
# 
# and you can download any file by appending hte zip archive name, e.g. 
# 
# [`https://s3.waw2-1.cloudferro.com/swift/v1/AUTH_afccea586afd4ef3bb11fe37dd1ddfbf/OrbitalAI_Data/335040-5955400_32618_2019-10-12T16-28-28.zip`](https://s3.waw2-1.cloudferro.com/swift/v1/AUTH_afccea586afd4ef3bb11fe37dd1ddfbf/OrbitalAI_Data/335040-5955400_32618_2019-10-12T16-28-28.zip)

# In[4]:


phisat2_locations = gpd.read_file("phisat2-locations.gpkg")


# In[5]:


len(phisat2_locations)


# In[6]:


fig, ax = plt.subplots(figsize=(15, 10))
WORLD_GDF.plot(ax=ax, color="b", alpha=0.2)
WORLD_GDF.boundary.plot(ax=ax, color="b")
phisat2_locations.plot(ax=ax, color="r");


# In[7]:


phisat2_locations.head(5)


# In[8]:


fig, ax = plt.subplots(figsize=(20, 5), ncols=3)
phisat2_locations.intersection_ratio.hist(ax=ax[0], bins=20, log=True)
ax[0].set_title("Overlap between Φ-sat-2 bbox and S2 image")
ax[0].set_xlabel("Overlap fraction")
ax[0].set_ylabel("log (# locations)")

phisat2_locations.time_difference.hist(ax=ax[1], bins=20)
ax[1].set_title("Time difference between S2 and PRISMA")
ax[1].set_xlabel("# days")
ax[1].set_ylabel("# locations")

phisat2_locations.cloud_cover_s2.hist(ax=ax[2], bins=20)
ax[2].set_title("Cloud cover of S2 images")
ax[2].set_xlabel("maxcc")
ax[2].set_ylabel("# locations");


# The simulation workflow can be parallelized over the locations to speed up the processing. `EOExecutor` allows to scale the processing over multiple processors/instances. For this to work we need to prepare the list of arguments for the executor, and the number of processing units to parallelize the execution over.

# In[49]:


exec_args = [
    {
        nodes[0]: {
            "bbox": get_utm_bbox(coords.y, coords.x),
            "time_interval": (atime.split("T")[0], atime.split("T")[0]),
        }
    }
    for atime, coords in zip(phisat2_locations.datetime_s2, phisat2_locations.geometry)
]


# In[67]:


exec_args[0]


# In[65]:


eoexecutor = EOExecutor(workflow=workflow, execution_kwargs=exec_args, save_logs=True)


# In[ ]:


eoexecutor.run(workers=4);

