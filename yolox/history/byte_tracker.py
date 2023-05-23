from re import I
import torch
import numpy as np
import torch.nn.functional as F

from yolox.tracker import matching
from .kalman_filter import KalmanFilter
from .basetrack import BaseTrack, TrackState


class STrack_history(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, srate):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self._z = self.tlwh_to_z(self._tlwh, self.image_height)
        self._layer = 0
        self._move_vector = [0, 0, 0, 0]
        self.history_list = []

        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.hidden, self.cell = torch.zeros(32).cuda(), torch.zeros(32).cuda()
        self.feature = torch.zeros(32).cuda()
        self.temporal = torch.ones(1).cuda() * srate
        self.delta_t = torch.FloatTensor([srate]).cuda()
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.world_frame_id = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_make_features(features, iou_graph, occulued_graph, direction_graph, history_tracker, stracks):
        def normalize_features(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = np.diag(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx
        features = normalize_features(features)
        features = torch.FloatTensor(features).cuda()
        iou_graph = torch.FloatTensor(iou_graph).cuda() #.unsqueeze(-1)
        # occulued_graph = torch.FloatTensor(occulued_graph).cuda().unsqueeze(-1)
        # direction_graph = torch.FloatTensor(direction_graph).cuda()
        # graph = torch.cat([iou_graph, occulued_graph, direction_graph], dim=-1)
        with torch.no_grad():
            # relation = history_tracker.relation_encoder(graph).squeeze()
            # relation_weight = torch.softmax(relation, dim=1)
            features = history_tracker.map_func(features)
            graph_features = history_tracker.graph_encoder(features, iou_graph)
            for i, graph_feature in enumerate(graph_features):
                stracks[i].feature = graph_feature

    def activate(self, kalman_filter, history_filter, frame_id):
        '''Record History'''
        _tlwh = self._tlwh # x y w h
        self.history_list.append({"frame_id": frame_id,
                                  "detection": self.tlwh_to_tlbr(_tlwh)})
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(_tlwh))

        if history_filter:
            # with torch.no_grad():
            #     hp, cp = history_filter.history_encoder(self.feature.unsqueeze(0), self.delta_t, (self.hidden.unsqueeze(0), self.cell.unsqueeze(0)))
            # self.hidden, self.cell = hp.squeeze(), cp.squeeze()
            with torch.no_grad():
                features = self.feature.unsqueeze(0).unsqueeze(0)
                mask = torch.BoolTensor([False]).cuda().unsqueeze(0)
                temporal = self.temporal.unsqueeze(0).unsqueeze(0)
                self.hidden = history_filter.history_encoder(features, mask).squeeze()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, history_filter, frame_id, new_id=False):
        '''Record History'''
        tlwh = new_track.tlwh # w y w h
        self.history_list.append({"frame_id": frame_id,
                                  "detection": self.tlwh_to_tlbr(tlwh)})
        """Reactive by a new tracklet"""
        self._move_vector = new_track.tlbr - self.tlbr
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(tlwh)
        )
        
        if history_filter:
            with torch.no_grad():
                features = new_track.feature.unsqueeze(0).unsqueeze(0)
                mask = torch.BoolTensor([False]).cuda().unsqueeze(0)
                temporal = self.temporal.unsqueeze(0).unsqueeze(0)
                self.hidden = history_filter.history_encoder(features, mask).squeeze()
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, history_filter, frame_id):
        '''Record History'''
        new_tlwh = new_track.tlwh # x y w h
        self.history_list.append({"frame_id": frame_id,
                                  "detection": self.tlwh_to_tlbr(new_tlwh)})
        """Reactive by a new tracklet"""
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self._move_vector = new_track.tlbr - self.tlbr
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))

        if history_filter:
            # with torch.no_grad():
            #     hp, cp = history_filter.history_encoder(new_track.feature.unsqueeze(0), self.delta_t, (self.hidden.unsqueeze(0), self.cell.unsqueeze(0)))
            # self.hidden, self.cell = hp.squeeze(), cp.squeeze()
            with torch.no_grad():
                if len(self.feature.shape) != 1:
                    # print(self.feature.shape, new_track.feature.shape)
                    self.feature = torch.cat((self.feature.squeeze(0), new_track.feature.unsqueeze(0)), dim=0).unsqueeze(0)
                else:
                    self.feature = torch.cat((self.feature.unsqueeze(0), new_track.feature.unsqueeze(0)), dim=0).unsqueeze(0)
                if len(self.temporal.shape) != 1:
                    # print(self.temporal.shape, new_track.temporal.shape)
                    self.temporal = torch.cat((self.temporal.squeeze(0), new_track.temporal.unsqueeze(0)), dim=0).unsqueeze(0)
                else:
                    self.temporal = torch.cat((self.temporal.unsqueeze(0), new_track.temporal.unsqueeze(0)), dim=0).unsqueeze(0)
                mask = torch.BoolTensor([False] * self.feature.shape[1]).cuda().unsqueeze(0)
                self.hidden = history_filter.history_encoder(self.feature, mask).squeeze()[-1, :]
                # self.feature = new_track.feature.clone() # self.feature.squeeze()
        
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    # @jit(nopython=True)
    def dtlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.history_list[-1]["detection"].copy()
        return ret
    
    @property
    # @jit(nopython=True)
    def observation(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        x_y_h = np.array([ret[0] + ret[2]//2, ret[1] + ret[3], ret[3]])       
        return x_y_h

    @property
    def _history(self):
        history = []
        for history_info in self.history_list:
            history.append(history_info['detection'])
        history = np.ascontiguousarray(history)
        # print(history)
        return history

    @property
    def history(self):
        world_frame_id = self.world_frame_id
        frame_id = self.frame_id
        # print(frame_id, world_frame_id)
        for prediction_id in range(frame_id, world_frame_id):
            history = self._history
            history_num = history.shape[0]
            if history_num == 1:
                return history[-1]
            X1 = history[:history_num - 1, :].T
            X2 = history[1:history_num, :].T
            rank = 2
            _, eigval, Phi = matching.dmd(X1, X2, rank)
            predict_detection = (Phi @ np.diag(eigval) @ (np.linalg.pinv(Phi) @ history[history_num-1, :])).real
            if np.isnan(predict_detection).any() or np.isinf(predict_detection).any():
                predict_detection = history[-1, :]
            if prediction_id != world_frame_id - 1:
                self.history_list.append({'frame_id': prediction_id,
                                          'detection': predict_detection.tolist()})
            else:
                return predict_detection

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret
    
    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_z(tlwh, image_height):
        ret = np.asarray(tlwh).copy()
        z_location = ret[1] + ret[3] // 2
        
        return z_location

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class HistoryTracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        frame_rate = frame_rate // args.sampling_rate
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.srate = args.sampling_rate
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        """if args.use_history:
            config_file = args.config_file
            cfg = config_test(config_file)
            self.history_tracker = HROP(cfg)
            self.history_tracker.load_state_dict(torch.load(cfg['Test']['ckpt_file']))
            self.history_tracker = self.history_tracker.cuda()
            self.history_tracker.eval()
        else:
            self.history_tracker = None"""

    def update(self, output_results, img_info, img_size, frame_id):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        BaseTrack.image_height = img_h
        
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []
            
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        for strack in strack_pool:
            strack.world_frame_id = self.frame_id
        dists = matching.iou_distance(strack_pool, detections)
        
        """if self.args.use_history:
            detection_iou_graph = 1.0 - matching.iou_distance(detections + detections_second, detections + detections_second)
            adject_matrix = matching.normalize_adj(detection_iou_graph)
            detection_features, detection_occulued_graph, detection_direction_graph = matching.make_occulued_matrix(detections + detections_second, img_w, img_h)
            STrack.multi_make_features(detection_features, adject_matrix, detection_occulued_graph,
                                       detection_direction_graph, self.history_tracker, detections + detections_second)
            history_dists = matching.feature_distance(strack_pool, detections)
            new_dists = []
            if len(strack_pool) > 0:
                for strack_i, strack in enumerate(strack_pool):
                    track_length = strack.tracklet_len
                    # w, b = 4, 8
                    # weight = 1.0 - math.exp(-(track_length+b)/w) # 0.85
                    weight = 0.9 # matching.map_func(track_length)
                    new_dists.append(dists[strack_i, :] * weight + history_dists[strack_i, :] * (1.0 - weight))
            else:
                new_dists = dists
            dists = np.asarray(new_dists)"""

        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet],   self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det,   self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det,   self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det,   self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        ''' Deal with unconfirmed tracks, usually tracks with only one beginning frame '''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet],   self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks """
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter,   self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
