def quick_inspect(v: object, indent=0, max_indent=-1):
    """ Recursively printing python object types """
    if max_indent >= 0 and indent >= max_indent:
        return
    if hasattr(v, 'shape'):
        print('\t'*indent, type(v).__name__, v.shape)
    elif isinstance(v, list):
        print('\t'*indent, 'list len=', len(v))
        for vv in v:
            if hasattr(vv, '__len__'):
                quick_inspect(vv, indent+1, max_indent=max_indent)
    elif isinstance(v, dict):
        for k, vv in v.items():
            print('\t'*indent, k)
            quick_inspect(vv, indent+1, max_indent=max_indent)
    else:
        print('\t'*indent, type(v))
