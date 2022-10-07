from hvs_hsi_pytorch.utils import video_converter_utils

video_converter_utils.process_video(
    input_vid='/Users/phinease/DataspellProjects/HVS_INN_MC_Project/data/finger_data/session_001/P010001_V03_iHSI_T01_2022-07-28-16-16-54.mov',
    white='/Users/phinease/DataspellProjects/HVS_INN_MC_Project/data/finger_data/session_001/exported_white_2022-07-15-20-29-45.png',
    dark=None,
    output_vid=None,
    tau_white=30,
    gain_white=0,
    rho=0.88,
    tau_dark=30,
    gain_dark=0,
    method="spectral",
    mode="raw",
    sensor="15.7.16.7",
    lightsource="sunoptic_x450",
    mask=False,
    radius=None,
    write_frames=True,
    start_frame=268,
    step_frame=5,
    end_frame=600,
)
