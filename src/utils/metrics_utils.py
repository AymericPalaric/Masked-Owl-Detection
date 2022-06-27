import torch
import torchvision
from src.constants import Detection

def compute_mAP(ground_truth_detections, detections, iou_threshold):
  precision, recall = compute_precision_recall(ground_truth_detections, detections, iou_threshold)
  auc = interpolated_auc(precision, recall)
  return sum(auc) / len(auc)

def compute_precision_recall(ground_truth_detections, detections, iou_threshold):
  """

  Args:
      ground_truth_detections (_type_): Tuple (img_idx, bbxs)
      detections (_type_): Tuple (img_idx, bbxs, scores)
      iou_threshold (_type_): _description_

  Returns:
      _type_: _description_
  """

  if not torch.any(ground_truth_detections) and torch.any(detections):
    precision = torch.tensor([1, 0, 0], dtype=torch.float32)
    recall = torch.tensor([0, 0, 1], dtype=torch.float32)
    return precision, recall
  if  torch.any(ground_truth_detections) and not torch.any(detections):
    precision = torch.tensor([1, 0, 0], dtype=torch.float32)
    recall = torch.tensor([0, 0, 1], dtype=torch.float32)
    return precision, recall

  _, indices = torch.sort(detections[Detection.SCORES], descending=True)

  bbxs = detections[Detection.BBXS][indices]
  im_idx = detections[Detection.IM_IDX][indices]

  g_bbxs = ground_truth_detections[Detection.BBXS]
  g_im_idx = ground_truth_detections[Detection.IM_IDX]

  matched = torch.zeros(g_bbxs.size(0), dtype=torch.bool)
  indices = torch.arange(g_bbxs.size(0))

  tp = torch.zeros((bbxs.size(0)), dtype=torch.int32)
  fp = torch.zeros((bbxs.size(0)), dtype=torch.int32)

  ious = torchvision.ops.box_iou(bbxs, g_bbxs)

  for d_idx in range(bbxs.size(0)):

    im_mask = (g_im_idx == im_idx[d_idx])

    if not torch.any(im_mask):
      fp[d_idx] = 1
      continue

    max_iou, max_idx = torch.max(ious[d_idx, im_mask], dim=0)
    max_idx = indices[im_mask][max_idx]

    if max_iou > iou_threshold:
      if matched[max_idx] == False:
        tp[d_idx] = 1
        matched[max_idx] = True
      else:
        fp[d_idx] = 1
    else:
      fp[d_idx] = 1

  tp_cumsum = torch.cumsum(tp, dim=0)
  fp_cumsum = torch.cumsum(fp, dim=0)

  precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
  recall = tp_cumsum / (g_bbxs.size(0) + 1e-6)

  precision = torch.cat((torch.tensor([1], dtype=torch.float32), precision, torch.tensor([0], dtype=torch.float32)))
  recall = torch.cat((torch.tensor([0], dtype=torch.float32), recall, torch.tensor([1], dtype=torch.float32)))

  precision = smooth_precision(precision)

  return precision, recall

def interpolated_auc(precision, recall):
    return torch.trapezoid(precision, recall)

def smooth_precision(precision):
    return torch.flip(torch.cummax(torch.flip(precision, dims=(0,)), dim=0)[0], dims=(0,))
