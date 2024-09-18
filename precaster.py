import torch

def precast_hook(trigger:torch.nn.Module, target:torch.nn.Module, **kwargs):
    def _hook(module, call_args):
        def pc(m):
            if (method := getattr(m, 'prep_cache', None)): method(*call_args, **kwargs)
        target.apply(pc)
    return trigger.register_forward_pre_hook(_hook)

def cleanup_hook(trigger:torch.nn.Module, target:torch.nn.Module):
    def _hook(module, call_args, result):
        def pc(m):
            if (method := getattr(target, 'dump_cache', None)): method()
        target.apply(pc)
    return trigger.register_forward_hook(_hook, always_call=True)

def add_precasting_to_flux_diffusion_model(diff_model, **kwargs):
    all_layers = [ db for db in diff_model.double_blocks ] + [ sb for sb in diff_model.single_blocks ]
    def activate(m):
        if hasattr(m,"use_cache"): m.use_cache = True
    for i, layer in enumerate(all_layers):
        layer.apply(activate)
        if i>0: precast_hook(trigger=previous, target=layer, **kwargs)
        cleanup_hook(trigger=layer, target=layer)
        if i>0: cleanup_hook(trigger=layer, target=previous) # extra care in case the cache happened too slowly, so got cleared before created!
        previous = layer
