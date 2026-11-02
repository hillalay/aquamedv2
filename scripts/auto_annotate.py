#!/usr/bin/env python3
"""Aquamed: video -> YOLO/BoT-SORT -> preview, CSV and CVAT XML.

Run from the repository: python scripts/auto_annotate.py --limit 1
--limit applies separately to each folder. Omit it to process all videos.
Folder labels are provisional, not model-predicted or human-reviewed truth.
Only karma and sakindurus are processed by default. Original videos are read-only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}
ACTIVITIES = 'unknown\nsakin_durus\nserbest\nkurbagalama\nsirtustu\nkelebek\nother'


def draw_person(image, xyxy, tracker_id, score, activity, font_scale=1.0):
    """Compact single-line label and thin box; no large background panel."""
    import cv2
    height, width = image.shape[:2]
    scale = max(0.65, min(height, width) / 720.0 * 0.9) * font_scale
    thickness = max(1, round(scale * 1.3))
    box_thickness = max(1, round(min(height, width) / 720.0))
    margin = max(3, round(min(height, width) / 240.0))
    color = (0, 230, 0) if activity == 'sakin_durus' else (0, 220, 255)
    a, b, c, d = map(round, xyxy)
    cv2.rectangle(image, (a, b), (c, d), color, box_thickness)
    identity = f'ID {tracker_id}' if tracker_id is not None else 'ID ?'
    label = activity if activity != 'unknown' else 'person'
    text = f'{identity} {label} | det {score:.2f}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    if tw + 2 * margin > width:
        scale *= max(0.05, (width - 2 * margin) / tw)
        thickness = max(1, round(scale * 1.3))
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(margin, min(a, width - tw - margin))
    y = b - margin - baseline
    if y - th < margin:
        y = b + margin + th
    y = max(th + margin, min(y, height - baseline - margin))
    # A small black outline keeps the text legible against water without a panel.
    cv2.putText(image, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def redraw_previews(args):
    """Re-render saved CSV boxes on original frames without loading YOLO/torch."""
    import cv2
    auto_root = args.root / 'data' / 'annotations' / 'auto'
    if args.redraw_run == 'latest':
        runs = sorted((p for p in auto_root.iterdir()
                       if p.is_dir() and any(p.rglob('summary.json'))), key=lambda p: p.name)
        if not runs:
            raise FileNotFoundError('Tamamlanmis video ciktisi bulunamadi.')
        run = runs[-1]
    else:
        if Path(args.redraw_run).name != args.redraw_run:
            raise ValueError('--redraw-run yalnizca run klasor adi veya latest olmali.')
        run = auto_root / args.redraw_run
    summaries = sorted(p for p in run.rglob('summary.json')
                       if p.parent.parent.name in args.folders)
    if not summaries:
        raise FileNotFoundError(f'Summary dosyasi bulunamadi: {run}')
    output = args.root / 'outputs' / 'previews' / (run.name + '_readable_' + datetime.now().strftime('%H%M%S_%f'))
    print(f'YOLO yeniden calismiyor. Kaydedilmis CSV kullaniliyor. Run: {run.name}', flush=True)
    for index, summary_path in enumerate(summaries, 1):
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        source = args.root / summary['source_relative']
        csv_path = summary_path.parent / 'detections.csv'
        destination = output / summary['folder'] / (summary_path.parent.name + '.mp4')
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.stem + '.partial.mp4')
        cap = cv2.VideoCapture(str(source))
        writer = None
        try:
            if not cap.isOpened():
                raise RuntimeError(f'Video acilamadi: {source}')
            writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*'mp4v'),
                                     summary['fps'], (summary['width'], summary['height']))
            if not writer.isOpened():
                raise RuntimeError('Onizleme yazicisi acilamadi.')
            with csv_path.open(newline='', encoding='utf-8-sig') as handle:
                groups = itertools.groupby(csv.DictReader(handle), key=lambda row: int(row['frame_index']))
                group_frame, rows = next(groups, (None, []))
                for frame_index in range(summary['frames_decoded']):
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f'Video erken bitti, kare: {frame_index}')
                    if frame.shape[1] != summary['width'] or frame.shape[0] != summary['height']:
                        raise RuntimeError('Video boyutu CSV kaynagiyla eslesmiyor.')
                    if group_frame is not None and group_frame < frame_index:
                        raise ValueError('CSV kare sirasi gecersiz.')
                    if group_frame == frame_index:
                        for row in rows:
                            draw_person(frame, [float(row[k]) for k in ('x1', 'y1', 'x2', 'y2')],
                                        int(row['tracker_id']) if row['tracker_id'] else None,
                                        float(row['person_confidence']), row['activity'], args.font_scale)
                        group_frame, rows = next(groups, (None, []))
                    writer.write(frame)
                if group_frame is not None:
                    raise ValueError('CSV video kare sayisini asiyor.')
            writer.release()
            writer = None
            partial.replace(destination)
            print(f'[{index}/{len(summaries)}] {source.name}\n  VIDEO: {destination}', flush=True)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
    print(f'Yeniden cizim tamamlandi: {output}', flush=True)
    return 0


def discover(root, folders, limit=0):
    jobs = []
    for folder in folders:
        directory = root / 'data' / 'raw' / folder
        if not directory.is_dir():
            raise FileNotFoundError(f'Klasor bulunamadi: {directory}')
        videos = sorted((p for p in directory.rglob('*')
                         if p.is_file() and p.suffix.lower() in EXTENSIONS),
                        key=lambda p: p.as_posix().casefold())
        if limit:
            videos = videos[:limit]
        print(f'{folder}: {len(videos)} video secildi.', flush=True)
        jobs.extend((folder, video) for video in videos)
    if not jobs:
        raise ValueError('Secilen klasorlerde video bulunamadi.')
    return jobs


def activity_for(folder):
    return 'sakin_durus' if folder == 'sakindurus' else 'unknown'


def output_key(video, root):
    relative = video.relative_to(root).as_posix()
    digest = hashlib.sha256(relative.encode('utf-8')).hexdigest()[:8]
    # Keep Windows paths short and avoid Unicode issues in generated outputs.
    return f'video_{digest}'


class CvatTracks:
    """CVAT IDs are local XML IDs; CSV retains the original tracker ID too."""
    def __init__(self, activity):
        self.activity = activity
        self.tracks = {}

    def add(self, frame, tracker_id, detection_index, xyxy):
        key = ('track', tracker_id) if tracker_id is not None else ('single', frame, detection_index)
        if key not in self.tracks:
            self.tracks[key] = {'id': len(self.tracks), 'boxes': []}
        item = self.tracks[key]
        item['boxes'].append((frame, tuple(float(v) for v in xyxy)))
        return item['id']

    def write(self, path, video_name, frame_count, width, height):
        root = ET.Element('annotations')
        ET.SubElement(root, 'version').text = '1.1'
        task = ET.SubElement(ET.SubElement(root, 'meta'), 'task')
        for key, value in [('id', 0), ('name', video_name), ('size', frame_count),
                           ('mode', 'interpolation'), ('overlap', 0),
                           ('start_frame', 0), ('stop_frame', frame_count - 1),
                           ('frame_filter', '')]:
            ET.SubElement(task, key).text = str(value)
        original = ET.SubElement(task, 'original_size')
        ET.SubElement(original, 'width').text = str(width)
        ET.SubElement(original, 'height').text = str(height)
        label = ET.SubElement(ET.SubElement(task, 'labels'), 'label')
        ET.SubElement(label, 'name').text = 'person'
        attrs = ET.SubElement(label, 'attributes')
        definitions = [('activity', self.activity, ACTIVITIES),
                       ('activity_source', 'folder' if self.activity != 'unknown' else 'unset',
                        'unset\nfolder\nmanual'),
                       ('review_status', 'unreviewed', 'unreviewed\nreviewed')]
        for name, default, values in definitions:
            attr = ET.SubElement(attrs, 'attribute')
            for key, value in [('name', name), ('mutable', 'True'),
                               ('input_type', 'select'), ('default_value', default),
                               ('values', values)]:
                ET.SubElement(attr, key).text = value

        def add_box(track, frame, xyxy, outside):
            x1, y1, x2, y2 = xyxy
            box = ET.SubElement(track, 'box', frame=str(frame), outside=str(int(outside)),
                                occluded='0', keyframe='1', z_order='0',
                                xtl=f'{x1:.3f}', ytl=f'{y1:.3f}',
                                xbr=f'{x2:.3f}', ybr=f'{y2:.3f}')
            for name, default, _ in definitions:
                ET.SubElement(box, 'attribute', name=name).text = default

        for item in self.tracks.values():
            track = ET.SubElement(root, 'track', id=str(item['id']), label='person', source='auto')
            boxes = item['boxes']
            for index, (frame, xyxy) in enumerate(boxes):
                if index and frame > boxes[index - 1][0] + 1:
                    last_frame, last_box = boxes[index - 1]
                    add_box(track, last_frame + 1, last_box, True)
                add_box(track, frame, xyxy, False)
            last_frame, last_box = boxes[-1]
            if last_frame + 1 < frame_count:
                add_box(track, last_frame + 1, last_box, True)
        ET.indent(root, space='  ')
        ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def process_video(video, folder, args, run_id):
    import cv2
    from ultralytics import YOLO

    key = output_key(video, args.root)
    ann_dir = args.root / 'data' / 'annotations' / 'auto' / run_id / folder / key
    preview_dir = args.root / 'outputs' / 'previews' / run_id / folder
    ann_dir.mkdir(parents=True, exist_ok=False)
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f'{key}.mp4'
    activity = activity_for(folder)
    cap = cv2.VideoCapture(str(video))
    writer = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Video acilamadi: {video}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError('Gecerli FPS okunamadi; zaman etiketleri uydurulmayacak.')
        expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Ilk kare okunamadi.')
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError('Onizleme videosu yazicisi acilamadi.')
        # A fresh model/tracker for every video prevents cross-video ID carryover.
        model = YOLO(args.model)
        tracks = CvatTracks(activity)
        frames = detections = untracked = frames_with_person = 0
        with (ann_dir / 'detections.csv').open('w', newline='', encoding='utf-8-sig') as csvfile:
            out = csv.writer(csvfile)
            out.writerow(['frame_index', 'time_sec', 'tracker_id', 'cvat_track_id',
                          'x1', 'y1', 'x2', 'y2', 'person_confidence',
                          'activity', 'activity_source', 'review_status'])
            while ok:
                result = model.track(frame, persist=True, tracker=args.tracker,
                                     classes=[0], device=args.device, imgsz=args.imgsz,
                                     conf=args.conf, verbose=False)[0]
                boxes = result.boxes
                annotated = frame.copy()
                if boxes is not None and len(boxes):
                    frames_with_person += 1
                    coords = boxes.xyxy.cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()
                    ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(coords)
                    for index, (xyxy, score, track_id) in enumerate(zip(coords, confidences, ids)):
                        track_id = int(track_id) if track_id is not None else None
                        x1, y1, x2, y2 = xyxy
                        xyxy = (max(0, min(width, x1)), max(0, min(height, y1)),
                                max(0, min(width, x2)), max(0, min(height, y2)))
                        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                            continue
                        xml_id = tracks.add(frames, track_id, index, xyxy)
                        out.writerow([frames, f'{frames / fps:.6f}',
                                      '' if track_id is None else track_id, xml_id,
                                      *[f'{v:.3f}' for v in xyxy], f'{score:.6f}',
                                      activity, 'folder' if folder == 'sakindurus' else 'unset',
                                      'unreviewed'])
                        detections += 1
                        untracked += int(track_id is None)
                        draw_person(annotated, xyxy, track_id, score, activity, args.font_scale)
                writer.write(annotated)
                frames += 1
                if frames % 100 == 0:
                    print(f'  {frames}/{expected_frames or "?"} kare', flush=True)
                ok, frame = cap.read()
        writer.release()
        writer = None
        tracks.write(ann_dir / 'annotations.xml', video.name, frames, width, height)
        summary = {'source_video': str(video), 'source_relative': video.relative_to(args.root).as_posix(),
                   'folder': folder, 'activity': activity,
                   'activity_source': 'folder' if folder == 'sakindurus' else 'unset',
                   'review_status': 'unreviewed', 'frames_decoded': frames,
                   'frames_reported': expected_frames, 'fps': fps,
                   'width': width, 'height': height, 'detections': detections,
                   'untracked_detections': untracked, 'frames_with_person': frames_with_person,
                   'preview': str(preview_path), 'model': args.model,
                   'tracker': args.tracker, 'device': args.device,
                   'imgsz': args.imgsz, 'conf': args.conf,
                   'note': 'Folder labels are provisional. No drowning/normal prediction. Preview has no audio.'}
        (ann_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        if expected_frames > 0 and frames != expected_frames:
            print(f'  UYARI: {expected_frames} bildirildi, {frames} kare okundu. Videoyu kontrol et.')
        print(f'  VIDEO: {preview_path}\n  CVAT: {ann_dir / "annotations.xml"}', flush=True)
        return summary
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--folders', nargs='+', choices=['karma', 'sakindurus'], default=['karma', 'sakindurus'])
    parser.add_argument('--limit', type=int, default=0, help='Video limit per folder; 0 means all')
    parser.add_argument('--model', default='yolov8s.pt')
    parser.add_argument('--tracker', default='botsort.yaml')
    parser.add_argument('--device', default='0')
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--conf', type=float, default=0.15)
    parser.add_argument('--list-only', action='store_true', help='List real paths without running YOLO')
    parser.add_argument('--font-scale', type=float, default=1.0, help='Label size multiplier; e.g. 1.5')
    parser.add_argument('--redraw-run', help='Re-render saved CSV only: run folder name or latest')
    args = parser.parse_args()
    args.root = args.root.resolve()
    if (args.limit < 0 or args.imgsz <= 0 or not 0 < args.conf < 1
            or not math.isfinite(args.font_scale) or not 0 < args.font_scale <= 5):
        parser.error('limit >= 0, imgsz > 0, 0 < conf < 1, 0 < font-scale <= 5 olmali.')
    if args.redraw_run:
        return redraw_previews(args)
    jobs = discover(args.root, args.folders, args.limit)
    for folder, path in jobs:
        print(f'  {folder}: {path.name} -> {activity_for(folder)}')
    if args.list_only:
        return 0
    import torch
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA kullanilamiyor. GPU kurulumunu kontrol et veya --device cpu kullan.')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    failures = []
    for index, (folder, video) in enumerate(jobs, 1):
        print(f'\n[{index}/{len(jobs)}] {folder}/{video.name}', flush=True)
        try:
            process_video(video, folder, args, run_id)
        except Exception as exc:
            failures.append({'video': str(video), 'error': str(exc)})
            print(f'  HATA: {exc}', file=sys.stderr, flush=True)
    print(f'\nBitti: {len(jobs) - len(failures)} basarili, {len(failures)} hatali. Run: {run_id}')
    if failures:
        error_path = args.root / 'data' / 'annotations' / 'auto' / run_id / 'errors.json'
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8')
    return int(bool(failures))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        print(f'HATA: {exc}', file=sys.stderr)
        raise SystemExit(1)
