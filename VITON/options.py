"""Command-line options of the try-on pipeline, shared by the live demos and the offline tools."""


def add_pipeline_arguments(parser):
    parser.add_argument('--dp_score', type=float, default=0.5,
                        help='minimum DensePose detection score; weaker detections (posters, mannequins) are ignored')
    parser.add_argument('--body_shape', default=None, choices=['estimated', 'train', 'zero'],
                        help="body shape for the 3D body render: the customer's own estimate, as in training "
                             "(average), or all shape parameters zeroed. Default: estimated for upper-body "
                             "garments, train for full-body ones")
    parser.add_argument('--keep_garments_on_gpu', action='store_true',
                        help='keep all garment networks on the GPU for instant switching; only with plenty of GPU '
                             'memory, otherwise Windows slows every frame down')
    parser.add_argument('--keep_in_front', default='hands_head', choices=['hands_head', 'hands', 'none'],
                        help='body parts drawn in front of the garment (hands_head, hands only, or none)')
    parser.add_argument('--blend', default='soft', choices=['soft', 'original'],
                        help="garment edges: 'soft' (anti-aliased, no dark outline) or 'original' (the repository's "
                             "hard 50%% mask; with --keep_in_front none it draws exactly like the original code)")
    parser.add_argument('--alpha_gate', type=int, default=30,
                        help='pixels (in the 512 network crop) an upper-body garment may extend beyond the body; '
                             'removes the background ghost a network produces outside its training poses (0 = off)')
    parser.add_argument('--track', default='on', choices=['on', 'off'],
                        help='on: the garment stays on one customer; off: the nearest person in every frame (original)')
    parser.add_argument('--zone', default='0,1',
                        help='left,right fractions of the image width in which a new customer is accepted, e.g. '
                             '0.2,0.8 ignores people at the sides (with --track on)')
    parser.add_argument('--smooth', default='on', choices=['on', 'off'],
                        help="on: filter the body estimate and the crop over time (less jitter); off: original")
    parser.add_argument('--temporal', type=float, default=0.5,
                        help='0..1 strength of the flicker filter on the output of upper-body garment networks (0 = off)')
    parser.add_argument('--dp_mode', default='box', choices=['box', 'full'],
                        help="box: DensePose only inside the body box BEV found (about twice as fast); full: the "
                             "whole DensePose detector (original)")
    parser.add_argument('--fp16', default='on', choices=['on', 'off'],
                        help='half precision for DensePose and the garment network (faster, half the GPU memory)')
    parser.add_argument('--cover_worn', default='on', choices=['on', 'off'],
                        help="on: the garment grows over the customer's own top where it would show around it "
                             "(shoulders, sleeves, collar) and is not cut by the collar or the cloth around a "
                             "hand; nothing is painted on the customer; off: original")
    parser.add_argument('--original', action='store_true',
                        help='shortcut for the original pipeline behaviour: --track off --smooth off --temporal 0 '
                             '--dp_mode full --fp16 off --cover_worn off')


def processor_options(args):
    """Keyword arguments for VITON.frame_processor.FrameProcessor from parsed arguments."""
    left, right = (float(v) for v in args.zone.split(','))
    options = {'dp_score': args.dp_score, 'body_shape': args.body_shape,
               'keep_garments_on_gpu': args.keep_garments_on_gpu, 'keep_in_front': args.keep_in_front,
               'blend': args.blend, 'alpha_gate': args.alpha_gate, 'track': args.track == 'on',
               'zone': (left, right), 'smooth': args.smooth == 'on', 'temporal': args.temporal,
               'dp_mode': args.dp_mode, 'fp16': args.fp16 == 'on', 'cover_worn': args.cover_worn == 'on'}
    if args.original:
        options.update(track=False, smooth=False, temporal=0.0, dp_mode='full', fp16=False, cover_worn=False)
    return options
