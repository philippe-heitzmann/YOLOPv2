# YOLOPv2 Road Segmentation — Object Contour Behavior

## The Problem

When YOLOPv2 generates the drivable area (road) segmentation mask, objects on the road (cyclists, pedestrians, vehicles) get **carved out** of the mask. The mask wraps around the object, leaving a hole where the object stands — even though the road surface physically continues beneath them.

This means that for our pedestrian-on-road brake detection use case, a pedestrian standing on the road might have **zero overlap** with the road mask, because the model has carved them out of the drivable area.

## Why This Happens

This behavior is **by design** in the training data, not a bug.

### BDD100K Drivable Area Definition

YOLOPv2 was trained on the **BDD100K** dataset, which defines drivable area as:

- **Directly drivable**: the lane where the ego vehicle has right-of-way
- **Alternatively drivable**: adjacent lanes the vehicle could change into

Critically, the BDD100K annotation guidelines state:

> "A lane cannot be driven on if occupied."

This means the ground truth labels **exclude regions occupied by other road users** — cars, cyclists, pedestrians. The model learned to predict "where can I actually drive right now" rather than "where is the physical road surface." The same patch of asphalt can be labeled as drivable in one frame (empty) and non-drivable in the next (someone is standing there).

### Model Architecture

The segmentation head outputs a **2-class prediction** per pixel (background vs. drivable). The `driving_area_mask()` function in `utils/utils.py` applies `torch.max` (argmax) over the channel dimension — there is no confidence threshold, just a hard argmax decision. If the model is even slightly more confident that a pixel is "occupied/non-drivable," it gets classified as background.

```python
def driving_area_mask(seg=None):
    da_predict = seg[:, :, 12:372, :]
    da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=2, mode='bilinear')
    _, da_seg_mask = torch.max(da_seg_mask, 1)   # argmax — no threshold
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
    return da_seg_mask
```

No post-processing (morphological closing, dilation, etc.) is applied — the raw argmax output is used directly.

## Impact on Brake Detection

For our `brake_ped_road.py` pipeline, this creates a subtle issue:

1. YOLOv11 detects a pedestrian bbox on the road
2. YOLOPv2 generates a road mask — but **carves out the pedestrian from the mask**
3. `ped_intersects_road()` checks overlap between bbox and road mask
4. The overlap may be **reduced or zero** because the pedestrian-shaped hole shrinks the intersection

In practice, this doesn't always cause missed detections because:
- The carve-out is rarely pixel-perfect (some road pixels leak into the bbox)
- The pedestrian bbox is usually larger than the carve-out
- The effect is strongest for large, well-defined objects (cars, cyclists) and weaker for distant/small pedestrians

But it **does reduce the reliability** of the intersection check, especially for cyclists and large nearby pedestrians.

## Potential Mitigations

### Option 1: Morphological Closing (Recommended)

Apply `cv2.morphologyEx` with `MORPH_CLOSE` to fill small holes in the road mask after generation. This fills object-shaped carve-outs while preserving the overall mask boundary.

```python
# In yolopv2_road_mask(), after generating road_mask:
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
```

**Tunable parameter**: kernel size. Larger kernel fills bigger holes (cars) but may also fill gaps between road and sidewalk. Start with `(25, 25)` and adjust.

- Pros: Simple, fast, fills most object carve-outs
- Cons: May connect road mask to nearby non-road areas if kernel is too large

### Option 2: Convex Hull of Road Mask

Compute the convex hull of the road mask region, which by definition fills all interior holes.

```python
contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hull_mask = np.zeros_like(road_mask)
for cnt in contours:
    hull = cv2.convexHull(cnt)
    cv2.fillConvexPoly(hull_mask, hull, 1)
road_mask = hull_mask
```

- Pros: Fills all holes, robust
- Cons: May overextend road mask into non-road areas (buildings, sidewalks) on curved roads

### Option 3: Use Raw Logits with Threshold

Instead of argmax, use the raw softmax/logit scores and apply a **lower threshold** for the drivable class. This makes the model more "generous" about what it labels as road, reducing the carve-out effect.

```python
def driving_area_mask_soft(seg, threshold=0.3):
    da_predict = seg[:, :, 12:372, :]
    da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=2, mode='bilinear')
    # Use softmax + threshold instead of argmax
    probs = torch.softmax(da_seg_mask, dim=1)
    road_prob = probs[:, 1, :, :]  # class 1 = drivable
    da_seg_mask = (road_prob > threshold).int().squeeze().cpu().numpy()
    return da_seg_mask
```

**Tunable parameter**: `threshold` (0.0 to 1.0). Lower values = more permissive road mask. Default argmax is equivalent to threshold=0.5.

- Pros: Fine-grained control, no spatial distortion
- Cons: May introduce noise at mask edges if threshold is too low

### Option 4: Dilation Only

Simpler than morphological closing — just dilate the mask to expand it into carve-out regions.

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
road_mask = cv2.dilate(road_mask, kernel, iterations=1)
```

- Pros: Very simple, fills carve-outs
- Cons: Expands the mask in ALL directions, not just inward into holes

## Recommendation

For the brake detection pipeline specifically, **Option 3 (soft threshold)** is the most principled approach because:
- It directly addresses the root cause (argmax being too aggressive)
- It gives a tunable parameter (`threshold`) that can be optimized
- It doesn't distort the spatial shape of the mask like morphological operations

A threshold of **0.3–0.4** should recover most road pixels under objects while avoiding excessive false road detection. This can be validated by visual inspection on the Amsterdam test set.

If a simpler approach is preferred, **Option 1 (morphological closing with a 25x25 kernel)** is the fastest to implement and test.
