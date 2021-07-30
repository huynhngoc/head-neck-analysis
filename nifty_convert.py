import SimpleITK as sitk
import h5py
import os


def save_as_nifty(data, filename):
    img = sitk.GetImageFromArray(data)
    img.SetSpacing((1, 1, 1))

    sitk.WriteImage(img, f'{filename}.nii.gz')


filename = "../../headneck_3d_new.h5"


for fold in range(11):
    with h5py.File(filename, 'r') as f:
        images = f[f'fold_{fold}']['input'][:]
        targets = f[f'fold_{fold}']['target'][:]
        patients = f[f'fold_{fold}']['patient_idx'][:]
    for i, pid in enumerate(patients):
        ct = images[i][..., 0]
        pt = images[i][..., 1]
        target = targets[i][..., 0]

        save_as_nifty(ct, f'W:/nii/imagesTr/HNC_{pid:03d}_0000')
        save_as_nifty(ct, f'W:/nii/imagesTr/HNC_{pid:03d}_0001')
        save_as_nifty(target.astype(int),
                      f'W:/nii/labelsTr/HNC_{pid:03d}_0000')


for fold in range(11, 14):
    with h5py.File(filename, 'r') as f:
        images = f[f'fold_{fold}']['input'][:]
        targets = f[f'fold_{fold}']['target'][:]
        patients = f[f'fold_{fold}']['patient_idx'][:]
    for i, pid in enumerate(patients):
        ct = images[i][..., 0]
        pt = images[i][..., 1]
        target = targets[i][..., 0]

        save_as_nifty(ct, f'W:/nii/imagesTs/HNC_{pid:03d}_0000')
        save_as_nifty(ct, f'W:/nii/imagesTs/HNC_{pid:03d}_0001')
        save_as_nifty(target.astype(int),
                      f'W:/nii/labelsTs/HNC_{pid:03d}_0000')


files = os.listdir('W:/nii/imagesTr')
for f in files:
    if len(f) < len('hnc_000_0000.nii.gz'):
        os.rename('W:/nii/imagesTr/' + f,
                  'W:/nii/imagesTr/' + f[:4] + '0' + f[4:])
