import os
import numpy as np
import json
from scipy.interpolate import interp1d

folder = 'myoutputs/demo/output_json/test_video'     # 여기 폴더에 프레임별 json 파일 있다고 가정

# 파일명 순서를 반드시 보장 (예시: 000001.json ~ ...)
files = sorted([f for f in os.listdir(folder) if f.endswith('.json')])

motions = []
for fname in files:
    with open(os.path.join(folder, fname)) as f:
        motions.append(json.load(f))

def is_zero_motion(m):
    return all(x == 0.0 for x in m['betas'])

# flatten/restore 함수는 위 구조 코드 그대로 복붙(또는 필요시 맞게 커스터마이즈)
# 예시:
def flatten_motion(motion):
    flat = []
    # body_pose, lhand_pose, rhand_pose 등은 2차원 배열 flatten 필요
    for key in ['betas', 'root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose',
                'lhand_pose', 'rhand_pose', 'trans', 'focal', 'princpt', 'img_size_wh']:
        val = motion[key]
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
            flat.extend(np.array(val).flatten())
        else:
            flat.extend(val)
    flat.append(motion['pad_ratio'])
    return np.array(flat)

def unflatten_motion(flat):
    idx = 0
    lens = [10,3,63,3,3,3,45,45,3,2,2,2,1] # 위 순서별 길이
    keys = ['betas', 'root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose',
            'lhand_pose', 'rhand_pose', 'trans', 'focal', 'princpt', 'img_size_wh', 'pad_ratio']
    motion = {}
    for k, l in zip(keys, lens):
        if k in ['body_pose', 'lhand_pose', 'rhand_pose']:
            motion[k] = flat[idx:idx+l].reshape((-1,3)).tolist()
        elif l == 1:
            motion[k] = float(flat[idx])
        else:
            motion[k] = flat[idx:idx+l].tolist()
        idx += l
    return motion

flat_motions = np.array([flatten_motion(m) for m in motions])
valid_idxs = [i for i, m in enumerate(motions) if not is_zero_motion(m)]
valid_idx_np = np.array(valid_idxs)

# 선형 보간
interp_func = interp1d(valid_idx_np, flat_motions[valid_idx_np], kind='linear', axis=0, fill_value='extrapolate')
all_idx = np.arange(len(motions))
filled_flat_motions = interp_func(all_idx)
filled_motions = [unflatten_motion(fm) for fm in filled_flat_motions]

# 덮어쓰기 - 원래 파일명에 맞게 저장
for fname, data in zip(files, filled_motions):
    with open(os.path.join(folder, fname), 'w') as f:
        json.dump(data, f)
