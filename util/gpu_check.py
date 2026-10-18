import torch


def require_cuda():
    """Stop with a clear message when PyTorch cannot see a CUDA GPU; the demo is far too slow on CPU."""
    if not torch.cuda.is_available():
        raise SystemExit('No CUDA GPU visible to PyTorch (torch %s, built for CUDA %s). Install a CUDA build of '
                         'torch and check `nvidia-smi`.' % (torch.__version__, torch.version.cuda))
    props = torch.cuda.get_device_properties(0)
    print('GPU: %s, %.1f GB, torch %s, CUDA %s' % (props.name, props.total_memory / 2 ** 30,
                                                   torch.__version__, torch.version.cuda))


def find_module(obj, depth=2):
    # BEV/ROMP and detectron2 wrap their networks; look a couple of attributes deep for an nn.Module
    if isinstance(obj, torch.nn.Module):
        return obj
    if depth == 0:
        return None
    for value in vars(obj).values() if hasattr(obj, '__dict__') else []:
        found = find_module(value, depth - 1)
        if found is not None:
            return found
    return None


def module_device(obj):
    module = find_module(obj)
    if module is None:
        return 'unknown'
    param = next(module.parameters(), None)
    return str(param.device) if param is not None else 'unknown'


def report_devices(named_models):
    """Print where each model's weights live and warn about any that are not on the GPU."""
    for name, obj in named_models.items():
        device = module_device(obj)
        warning = '' if device.startswith('cuda') else '   <-- NOT ON GPU'
        print('  %-10s on %s%s' % (name, device, warning))
