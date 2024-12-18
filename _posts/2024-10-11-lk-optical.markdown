---
layout: post
title:  "Lucas Kanade Optical Flow"
categories: jekyll update
---


# Lucas-Kanade Optical Flow: Decoding Motion Dynamics

## Introduction

Optical flow is a fascinating concept in computer vision that tracks the apparent motion of objects between consecutive image frames. Developed by Bruce D. Lucas and Takeo Kanade in 1981, the Lucas-Kanade method provides an elegant solution for estimating pixel motion.

## Methodology

The Lucas-Kanade method assumes two critical constraints:
1. **Brightness Consistency**: Pixel intensities remain constant across frames
2. **Spatial Coherence**: Neighboring pixels move similarly

The fundamental optical flow equation:

```
Ix * u + Iy * v + It = 0
```

Where:
- `Ix`: Spatial x-gradient
- `Iy`: Spatial y-gradient
- `It`: Temporal gradient
- `u`: Horizontal velocity
- `v`: Vertical velocity

## Intuitive Mathematical Approach

Unlike complex tracking algorithms, Lucas-Kanade uses a simple approach:
- Compute local image gradients
- Solve a least-squares problem for motion vectors
- Estimate velocity for small pixel neighborhoods

## Python Implementation

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

class OpticalFlowTracker:
    def __init__(self, max_corners=100, quality_level=0.3, min_distance=7):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    
    def track_features(self, prev_frame, next_frame, prev_points):
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_frame, next_frame, 
            prev_points, None, **self.lk_params
        )
        
        good_new = next_points[status == 1]
        good_old = prev_points[status == 1]
        
        return good_new, good_old
    
    def detect_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(
            gray, 
            maxCorners=self.max_corners, 
            qualityLevel=self.quality_level, 
            minDistance=self.min_distance
        )
        
        return corners
    
    def visualize_flow(self, frame, prev_points, next_points):
        mask = np.zeros_like(frame)
        
        for (new, old) in zip(next_points, prev_points):
            a, b = new.ravel()
            c, d = old.ravel()
            
            mask = cv2.line(
                mask, 
                (int(a), int(b)), 
                (int(c), int(d)), 
                (0, 255, 0), 2
            )
            mask = cv2.circle(
                mask, 
                (int(a), int(b)), 
                5, (0, 0, 255), -1
            )
        
        output = cv2.add(frame, mask)
        return output
    
    def compute_motion_vectors(self, prev_points, next_points):
        motion_vectors = next_points - prev_points
        return motion_vectors

video_path = 'yoyo.mp4'
cap = cv2.VideoCapture(video_path)

tracker = OpticalFlowTracker()

ret, prev_frame = cap.read()
prev_points = tracker.detect_features(prev_frame)

while True:
	ret, frame = cap.read()
	if not ret:
		break
	
	next_points, prev_points = tracker.track_features(
		cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
		cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
		prev_points
	)
	
	flow_frame = tracker.visualize_flow(frame, prev_points, next_points)
	motion_vectors = tracker.compute_motion_vectors(prev_points, next_points)
	
	cv2.imshow('Optical flow ~~ ', flow_frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
	prev_frame = frame
	prev_points = next_points

cap.release()
cv2.destroyAllWindows()

```


There are some limitations of the method because it 
- Assumes small motion between frames
- Struggles with large displacements
- Sensitive to brightness changes

## Other Awesome Alternatives

- Farneback Method
- FlowNet (Deep Learning)
- RAFT (Recurrent All Pairs Field Transforms) (This is a relatively recent optical flow method which learns a neural network for learning from data)


Importantly the Lucas-Kanade method has:
- Time Complexity: O(n * m)
- Space Complexity: O(n)
where
- n = number of features
- m = window size

## Conclusion

Lucas-Kanade optical flow represents a foundational technique in motion estimation, bridging mathematical elegance with practical computer vision applications.