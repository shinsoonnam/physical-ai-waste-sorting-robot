"""Physical AI: camera-based sorting demo for Google Colab (simulation).
AI: RandomForest RGB classification; planning: A* obstacle-aware grid navigation;
robot: simplified kinematic mobile carrier (not a physical grasping arm).
"""
import os, csv, heapq, math
import numpy as np
import pybullet as p
from sklearn.ensemble import RandomForestClassifier
import imageio.v2 as imageio
import cv2

OUT = 'recycling_outputs'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(23)
NAMES = ['PLASTIC', 'PAPER', 'METAL']
COLORS = np.array([[0.92, 0.22, 0.18], [0.20, 0.42, 0.91], [0.18, 0.80, 0.33]])
TARGETS = [(1.35, -0.95), (1.35, 0.0), (1.35, 0.95)]
# Train a supervised model on synthetic camera-like RGB measurements.
xx, yy = [], []
for label, rgb in enumerate(COLORS):
    a = np.clip(rng.normal(rgb * 205, 17, size=(360, 3)), 0, 255)
    xx.extend(a); yy.extend([label] * len(a))
model = RandomForestClassifier(n_estimators=90, random_state=12).fit(np.asarray(xx), yy)
print('AI model trained on synthetic RGB observations:', len(yy), 'samples')

p.connect(p.DIRECT)
p.setGravity(0, 0, -9.8)
p.setTimeStep(1/120)
def solid_box(pos, half, color, mass=0):
    c = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=list(color)+[1])
    return p.createMultiBody(mass, c, v, pos)
solid_box((0,0,-.06),(2.1,1.5,.06),(.86,.89,.91))
for k, goal in enumerate(TARGETS):
    solid_box((goal[0], goal[1], .015),(.36,.35,.015),COLORS[k]*.6)
# World: objects are distinguishable by their appearance (color + shape).
objects=[]
positions=[(-1.35,-1.00),(-1.35,0.0),(-1.35,1.0),(-.75,-1.),(-.75,0.),(-.75,1.)]
labels=[0,1,2,2,0,1]
for i,(xy,label) in enumerate(zip(positions,labels)):
    if label==1:
        c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.13,.10,.12])
        v = p.createVisualShape(p.GEOM_BOX, halfExtents=[.13,.10,.12], rgbaColor=list(COLORS[label])+[1])
    else:
        c = p.createCollisionShape(p.GEOM_CYLINDER, radius=.11, height=.25)
        v = p.createVisualShape(p.GEOM_CYLINDER, radius=.11, length=.25, rgbaColor=list(COLORS[label])+[1])
    bid=p.createMultiBody(0,c,v,[*xy,.16])
    objects.append((bid,label,xy))
# Simplified carrier robot: chassis + top marker, no true grasp physics.
robot=solid_box((-1.85,-1.25,.12),(.15,.19,.11),(.16,.16,.19))
marker=solid_box((-1.85,-1.25,.27),(.09,.10,.035),(.96,.76,.18))
view=p.computeViewMatrix(cameraEyePosition=[0,-.25,4.8],cameraTargetPosition=[0,0,0],cameraUpVector=[0,1,0])
proj=p.computeProjectionMatrixFOV(fov=51,aspect=480/360,nearVal=.1,farVal=10)
def capture():
    im=p.getCameraImage(480,360,viewMatrix=view,projectionMatrix=proj,renderer=p.ER_TINY_RENDERER)
    rgb=np.asarray(im[2],dtype=np.uint8).reshape(360,480,4)[:,:,:3]
    seg=np.asarray(im[4],dtype=np.int64).reshape(360,480)
    return rgb,seg
# Acquire actual rendered pixels for each object (segmentation IDs are for ROI only).
observations=[]
for body,truth,xy in objects:
    frame,seg=capture()
    mask=(seg & ((1<<24)-1))==body
    if mask.sum()<15:
        raise RuntimeError('Object not visible from the camera; check camera placement')
    feature=np.median(frame[mask],axis=0)
    pred=int(model.predict([feature])[0])
    confidence=float(model.predict_proba([feature]).max())
    observations.append((pred,confidence,feature))
print('Predictions:',[(NAMES[pred],round(conf,2)) for pred,conf,_ in observations])

# Grid A*: plan around all uncollected waste objects, with robot clearance.
# Robot chassis is 0.30 x 0.38 m; 0.36 m clearance also covers carried item.
GRID = 0.075
BOUNDS = (-2.02, 2.02, -1.38, 1.38)
CLEARANCE = 0.35
ROBOT_MARGIN = 0.20
NX = int(round((BOUNDS[1]-BOUNDS[0])/GRID))+1
NY = int(round((BOUNDS[3]-BOUNDS[2])/GRID))+1

def to_cell(xy):
    return (int(round((float(xy[0])-BOUNDS[0])/GRID)),
            int(round((float(xy[1])-BOUNDS[2])/GRID)))

def to_xy(cell):
    return np.array([BOUNDS[0]+cell[0]*GRID, BOUNDS[2]+cell[1]*GRID])

def astar_path(origin, destination, other_items):
    # To pick up an object, stop at its west side; never drive over its center.
    # No objects yet collected are considered obstacles.
    blocked=set()
    for ix in range(NX):
        for iy in range(NY):
            pos=to_xy((ix,iy))
            if (pos[0] < BOUNDS[0]+ROBOT_MARGIN or pos[0] > BOUNDS[1]-ROBOT_MARGIN or
                pos[1] < BOUNDS[2]+ROBOT_MARGIN or pos[1] > BOUNDS[3]-ROBOT_MARGIN or
                any(np.linalg.norm(pos-np.array(o)) < CLEARANCE for o in other_items)):
                blocked.add((ix,iy))
    source, dest=to_cell(origin),to_cell(destination)
    if source in blocked or dest in blocked:
        raise ValueError(f'Path endpoint blocked: {origin} -> {destination}; '
                         'adjust the pickup offset or clearance')
    moves=[(dx,dy,math.hypot(dx,dy)) for dx in (-1,0,1) for dy in (-1,0,1) if dx or dy]
    def h(c): return math.hypot(c[0]-dest[0],c[1]-dest[1])
    pq=[(h(source), 0.0, source)]
    prev={}
    best={source:0.0}
    while pq:
        _,g,current=heapq.heappop(pq)
        if g>best[current]+1e-9: continue
        if current==dest: break
        for dx,dy,cost in moves:
            nb=(current[0]+dx,current[1]+dy)
            if not(0<=nb[0]<NX and 0<=nb[1]<NY) or nb in blocked: continue
            # Prevent diagonal corner cutting around inflated object cells.
            if dx and dy and ((current[0]+dx,current[1]) in blocked or
                              (current[0],current[1]+dy) in blocked): continue
            ng=g+cost
            if ng < best.get(nb,1e100):
                best[nb]=ng;prev[nb]=current
                heapq.heappush(pq,(ng+h(nb),ng,nb))
    if dest!=source and dest not in prev:
        raise RuntimeError(f'No obstacle-free route: {origin} -> {destination}')
    cells=[dest]
    while cells[-1]!=source: cells.append(prev[cells[-1]])
    cells.reverse()
    # Keep exact start and goal, ensure interpolated segments stay safe.
    pts=[np.asarray(origin,dtype=float)]+[to_xy(c) for c in cells[1:-1]]+[np.asarray(destination,dtype=float)]
    return np.array(pts)

def move_along(points, fraction):
    points=np.asarray(points)
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    total=lengths.sum()
    if total<1e-10: return points[-1].copy()
    d=np.clip(fraction,0,1)*total
    for i,segment in enumerate(lengths):
        if d<=segment or i==len(lengths)-1:
            return points[i]+(points[i+1]-points[i])*(d/segment if segment else 0)
        d-=segment
    return points[-1].copy()

FPS=8
FRAMES_PER_ITEM=96  # 12 seconds per item; 72 seconds total
filename=os.path.join(OUT,'ai_recycling_robot_obstacle_avoidance.mp4')
log=[]
start=np.array([-1.82,-1.12],dtype=float)
# Earlier items have been delivered; every later item is an obstacle.
with imageio.get_writer(filename,fps=FPS,codec='libx264',quality=7,macro_block_size=16) as writer:
    for idx,(body,truth,item_xy) in enumerate(objects):
        pred,conf,features=observations[idx]
        target=TARGETS[pred]
        pickup=np.asarray(item_xy,dtype=float)+np.array([-.47,0.])
        drop=np.asarray(target,dtype=float)+np.array([-.43,0.])
        # Outbound: target is ALSO an obstacle; approach from a safe distance.
        obstacles_to_pickup=[o[2] for j,o in enumerate(objects) if j>=idx]
        approach=astar_path(start,pickup,obstacles_to_pickup)
        # Return: carried item is excluded but all waiting objects remain obstacles.
        obstacles_to_drop=[o[2] for j,o in enumerate(objects) if j>idx]
        outbound=astar_path(pickup,drop,obstacles_to_drop)
        print(f'Item {idx+1}: outbound {len(approach)} waypoints; return {len(outbound)} waypoints')
        for tick in range(FRAMES_PER_ITEM):
            t=tick/FRAMES_PER_ITEM
            if t<.35:
                xy=move_along(approach,t/.35); phase='AVOID WASTE / APPROACH'
            elif t<.43:
                xy=pickup; phase='PICK UP (SIMULATED)'
            elif t<.82:
                xy=move_along(outbound,(t-.43)/.39); phase='AVOID WASTE / TRANSPORT'
            elif t<.90:
                xy=drop; phase='DROP OFF (SIMULATED)'
            else:
                xy=drop; phase='NEXT ITEM'
            p.resetBasePositionAndOrientation(robot,[*xy,.12],[0,0,0,1])
            p.resetBasePositionAndOrientation(marker,[*xy,.27],[0,0,0,1])
            if .43<=t<.86:
                p.resetBasePositionAndOrientation(body,[xy[0]+.19,xy[1],.29],[0,0,0,1])
            elif t>=.86:
                place=[target[0]+.10*((idx//3)-.5),target[1]+.09*((idx%2)-.5),.16]
                p.resetBasePositionAndOrientation(body,place,[0,0,0,1])
            img,_=capture()
            canvas=cv2.copyMakeBorder(img,42,28,0,0,cv2.BORDER_CONSTANT,value=(22,28,36))
            cv2.putText(canvas,'AI RECYCLING ROBOT | AI + A* OBSTACLE AVOIDANCE',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
            status=f'Item {idx+1}/6 | AI: {NAMES[pred]} ({conf:.0%}) | {phase}'
            cv2.putText(canvas,status,(10,382),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(canvas,'RED: PLASTIC  BLUE: PAPER  GREEN: METAL',(10,406),cv2.FONT_HERSHEY_SIMPLEX,.43,(210,220,235),1,cv2.LINE_AA)
            writer.append_data(canvas)
        start=drop.copy()
        log.append({'item':idx+1,'actual':NAMES[truth],'predicted':NAMES[pred],
                    'confidence':round(conf,3),'correct':bool(pred==truth),
                    'pickup_waypoints':len(approach),'return_waypoints':len(outbound)})
p.disconnect()
with open(os.path.join(OUT,'predictions.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
print('Saved:',filename,' duration seconds:',len(objects)*FRAMES_PER_ITEM/FPS)
print('Classification accuracy for the 6 placed objects:',sum(v['correct'] for v in log),'/',len(log))
