
USER_DIR="/Users/kazu-gor"
REMOTE_NAME="student"
REMOTE_IP="192.168.0.68"
REMOTE_DIR="/home/student/git/medical-image-processing/dataset"

# paramD=(0.625 0.615 0.62 0.63 0.635 0.55 0.575 0.6 0.65 0.675 0.7)
paramD=(0.615 0.62 0.63 0.635 0.55 0.575 0.6 0.65 0.675 0.7)

for d in ${paramD[@]}; do
    scp -r ${USER_DIR}/git/medical-image-processing/dataset/CrossValidation_original_crop_${d} ${REMOTE_NAME}@${REMOTE_IP}:${REMOTE_DIR}
done
