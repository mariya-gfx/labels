#!/usr/bin/env python3

"""Turn json ai semantic export into a transform."""

import argparse
import base64
import collections
import contextlib
import hashlib
import itertools
import json
import pprint
import re
import sys


PATH_CHAR = '\xb7'


def deriveColor(col):
    """Create CSS color from i9r json."""
    if not col:
        return None
    width = 2
    if all(map((lambda n: n >> 4 == n & 15), col.values())):
        col = {k: v & 15 for k, v in col.items()}
        width = 1
    return '#{red:0{w}x}{green:0{w}x}{blue:0{w}x}'.format(**col, w=width)


def fixColor(string):
    if string is None:
        return '#fff'
    if 'transp' in string:
        return 'transparent'
    return '#' + string.lstrip('#')


def deriveClass(url, defaults):
    for operator, existing, replace in defaults['rewrite_class']:
        if operator == '==' and url == existing:
            return replace
        elif operator == '*=' and url.startswith(existing):
            return replace + url[len(existing):]
    return url


def deriveHashIdent(obj):
    if obj['classFor'] == 'http://www.w3.org/ns/org#OrganisationalUnit':
        raw = obj['note']['org_unit_id']
    else:
        raw = str(obj['n']) + deriveIdent(obj['path'])
    hash_prefix = hashlib.sha256(raw.encode('utf-8')).digest()[:6]
    return base64.urlsafe_b64encode(hash_prefix).decode('ascii')


def deriveIdent(path):
    return '-'.join(path.split(PATH_CHAR)[1:]).strip()


def deriveNote(contents):
    # TODO: Parse a bit more like TOML rules, and fail sensibly
    lines = (
        contents.strip()
        .replace('\u201c', '"')
        .replace('\u201d', '"')
        .replace('" ', '"\n')
        .replace('org_unit_id:', 'org_unit_id =')
        .replace(' class:', '\nclass =')
        .replace('fullname =', 'fullName =')
        .replace('title background', 'background')
        .splitlines())
    entries = (line.partition('=') for line in lines)
    return {k.strip(): v.strip().strip('"').strip() for k, _, v in entries}


def deriveQcon(path):
    parts = path.split(PATH_CHAR)
    if not parts or len(parts) < 2:
        return ''
    off = 1 if len(parts) < 3 else 2
    return ': '.join(p.strip() for p in parts[off:])


def deriveAcon(obj):
    if 'parent' not in obj:
        return ''
    if obj['classFor'] == 'http://www.w3.org/ns/org#OrganisationalUnit':
        return 'orgunit-{:d}'.format(int(obj['note']['org_unit_id']))
    parent = obj['parent']
    while True:
        anchor = parent.get('anchorName')
        if anchor != 'Management':
            break
        parent = parent['parent']
    local = obj['note'].get('fullName') or ' '.join(obj['text'].split())
    if anchor:
        return ': '.join([anchor, local])
    return local


def deriveText(string):
    lines = string.replace('\u0003', '\r').strip().split('\r')
    return '\n'.join(line.strip() for line in lines)


def deriveLocation(scaler, loc):
    if loc is None:
        return None
    if isinstance(loc, list):
        result = pair(scaler.latlng_bounds(loc))
    elif isinstance(loc, dict):
        result = scaler.path(loc)
    else:
        result = [scaler.path(p) for d in loc for p in d]
        if not result:
            return None
    return repr(result).replace(' ', '')


def pair(alist):
    return [[alist[i - 1], alist[i]] for i in range(1, len(alist), 2)]


def unparentable(o):
    return o['classFor'] == 'vm:Flow'


def json_str(obj):
    """Dump as json str in most compact representation"""
    return json.dumps(obj, separators=(',', ':'))


def resolve_layer(messy, name_for_layer):
    name = '-'.join(messy.lstrip('-~').split(' · ')[0].strip().lower().split())
    zoom = None
    if not name.strip('-0123456789'):
        name, zoom = '', name
    if name_for_layer is not None:
        # preserve existing zoom layers if present and not overridden by name_for_layer
        if '@' in name and '@' not in name_for_layer:
            zoom = name.split('@')[1]
        name = name_for_layer
    if zoom:
        return name + '@' + zoom
    return name


def map_layers(layers, name_for_layer, omit):
    for layer in layers:
        name = layer['name']
        if name.startswith(('~', '-', '-~')) and name not in omit:
            layer_name = resolve_layer(name, name_for_layer)
            yield dict(depth=0, uid=name, layer=layer_name, **layer)


def map_labels(labels, parent, counter, defaults):
    for label in labels:
        contents = label['name']['contents']
        if contents == parent.get('contents') or not contents.strip():
            print('DUPE', contents.replace('\r', ' '), file=sys.stderr)
            continue  # Skip duplicated immediate child labels
        note = deriveNote(label['name'].pop('note', ''))
        yield dict(
            depth=parent['depth'] + 1,
            children=label.get('children', ()),
            layer=parent['layer'],
            parent=parent,
            area=label.get('area'),
            dataLocation=label.get('dataCollections', {}).values(),
            visibleBounds=label.get('visibleBounds'),
            path=label['path'],
            uid=label['uID'],
            n=next(counter),
            text=deriveText(label['name'].pop('contents')),
            anchorName=note.pop('anchorName', parent.get('anchorName', '')),
            classFor=deriveClass(note.pop('class', parent.get('class', defaults['class'])), defaults),
            note=note,
            background=note.pop('background', label.get('background', None)),
            **label['name'],
        )


def make_lens_details(stack):
    layers_and_zooms = collections.defaultdict(set)
    for o in stack:
        # need to make some assumptions about layer name format in order to extract lensdetail details
        if not re.search(r"[a-zA-Z0-9]*@(?:[0-9]|[0-9]\-[0-9])$", o['layer']):
            raise ValueError("invalid layer name for generating forLensDetail")
        layername, zoom = o['layer'].split('@')
        layers_and_zooms[layername].add(zoom)

    lens_details = dict()
    for layer in layers_and_zooms:
        for i, zoom in enumerate(sorted(layers_and_zooms[layer])):
            layername_again = '{}@{}'.format(layer, zoom)
            detail_level_name = 'vm:_detail_{}_{}'.format(layer, i + 1)
            lens_details[layername_again] = detail_level_name

    return lens_details


def noop(*args, **kwargs):
    """Do nothing"""


def build_lens_details(defaults, stack):
    if not defaults['lens_detail']:
        return noop
    details = make_lens_details(stack)

    def apply_to_row(row, o):
        row['lensDetail'] = details[o['layer']]

    return apply_to_row


def iter_labels(layers, scaler, defaults):
    counter = itertools.count()
    stack = list(layers)
    maybe_apply_lens_detail = build_lens_details(defaults, stack)
    stack.reverse()
    while stack:
        o = stack.pop()
        stack.extend(map_labels(reversed(o['children']), o, counter, defaults))
        if o.get('type') or o.get('parent') is None:
            # This is not a label, it's probably a top level layer
            defaults['tracking'](o, None)
            continue
        row = dict(
            ident=deriveHashIdent(o),
            qcontents=deriveAcon(o),
            layer=o['layer'],
            pqcontents=deriveAcon(o['parent']),
            area=deriveLocation(scaler, o['area'] or o['visibleBounds']),
            dataLocation=deriveLocation(scaler, o['dataLocation']),
            type=o['classFor'],
        )
        if unparentable(o):
            del row['pqcontents']
        if not o.get('hidden'):
            display = dict(
                box=scaler.latlng_bounds(o['bounds']),
                background=o.get('backgroundColour') or fixColor(o['background']),
                color=deriveColor(o['characterColour']),
                text=o['text'],
                fontFamily=o['typeface'],
                fontSize=scaler.distance(o['fontSize']),
                # TODO: matrix, opacity
            )
            row['display'] = json_str(display)
        # Experiment with including point even for hidden labels
        row['geoPoint'] = json_str(scaler.latlng(o['centrePoint']))
        maybe_apply_lens_detail(row, o)
        defaults['tracking'](o, row)
        yield row


def track_label_details(o, row):
    """Default out of band output for processed labels."""
    name = o['name'] if row is None else o['text'].replace('\n', ' ')
    print(
        '{indent}{name}'.format(indent=o['depth'] * '  ', name=name),
        file=sys.stderr)


def _as_slug(text, _pat=re.compile(r'\W+')):
    """Make text suitable for appearing in an IRI.

    Same basic logic as used by sheet-to-triples.
    """
    return _pat.sub('-', text.replace('&', ' and ')).strip('-').lower()


@contextlib.contextmanager
def track_labels_tabular(outfile, individual_prefix):
    with open(outfile, 'w') as f:
        print('layer\tdepth\tlabel\tname\tparent\tanchor\tiri_suffix', file=f)

        def _track_label_as_tsv(o, row):
            if row is None:
                return
            record = '{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                o['layer'],
                o['depth'],
                '|'.join(o['text'].split('\n')),
                row['qcontents'],
                row['pqcontents'],
                o['anchorName'],
                individual_prefix + _as_slug(row['qcontents']),
            )
            print(record, file=f)

        yield _track_label_as_tsv


def build_tracker(args):
    """Create a function for reporting labels as they are processed."""
    if args.quiet:
        return contextlib.nullcontext(lambda o, row: None)
    if args.export_tabular is None:
        return contextlib.nullcontext(track_label_details)
    return track_labels_tabular(args.export_tabular, args.individual_prefix)


class Scaler:
    """Class to go from Illustrator coords system to Leaflet map coords."""

    ROUND = 2
    SIZE = 256

    def __init__(self, origin_x, origin_y, scale):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.scale = scale

    def __repr__(self):
        return '{}({}, {}, {})'.format(
            self.__class__.__name__, self.origin_x, self.origin_y, self.scale)

    @classmethod
    def from_rect(cls, x1, y1, x2, y2):
        scale_x = abs(cls.SIZE / (x2 - x1))
        scale_y = abs(cls.SIZE / (y2 - y1))
        if scale_x != scale_y:
            print('MISMATCH artboard', x1, y1, x2, y2, file=sys.stderr)
        return cls(x1, y1, scale_x)

    @classmethod
    def from_artboard(cls, ab):
        return cls.from_rect(ab['left'], ab['top'], ab['right'], ab['bottom'])

    def _to(self, n):
        return round(n * self.scale, self.ROUND)

    def path(self, o):
        if o.get('type') is not None:
            return [self.latlng(segment['anchor']) for segment in o['points']]
        if o.get('areaType') == 'pathPoints':
            return [self.latlng(segment['anchor']) for segment in o['points']]
        raise NotImplementedError(o)

    def distance(self, n):
        return self._to(n)

    def latlng(self, coord):
        x, y = coord
        return [self._to(y - self.origin_y), self._to(x - self.origin_x)]

    def latlng_bounds(self, coord):
        x1, y1, x2, y2 = coord
        return [
            self._to(y1 - self.origin_y),
            self._to(x1 - self.origin_x),
            self._to(y2 - self.origin_y),
            self._to(x2 - self.origin_x),
        ]


def to_transform(data, source, name_for_layer, omit, prefix, defaults):
    scaler = Scaler.from_artboard(data['artboard'])
    layers = map_layers(data['layers'], name_for_layer, omit)
    rows = list(iter_labels(layers, scaler, defaults))
    # TODO: merge labels across views
    transform = {
        'data': rows,
        'lets': {
            'iri': 'vm:l{row[layer].as_text}-{row[ident].as_text}',
            'content': '{row[qcontents].as_text}',
        },
        'triples': [
            ('{iri}', 'rdf:type', 'vm:Label'),
            ('{iri}', 'vm:atGeoPoint', '{row[geoPoint].as_text}'),
            ('{iri}', 'vm:display', '{row[display].as_text}'),
            ('{iri}', 'vm:forThing', '{for}'),
        ],
    }

    if defaults['lens_detail']:
        transform['lets']['lensDetail'] = '{row[lensDetail].as_text}'
        transform['triples'].append(
            ('{iri}', 'vm:forLensDetail', '{lensDetail}'),
        )
    else:
        transform['lets']['layer'] = '{row[layer].as_text}'
        transform['triples'].append(
            ('{iri}', 'vm:forLayer', '{layer}'),
        )

    if source:
        transform['lets']['for'] = prefix + '{row[qcontents].as_slug}'
        transform['triples'].extend([
            ('{for}', 'rdf:type', '{row[type].as_text}'),
            ('{for}', 'vm:name', '{row[qcontents].as_text}'),
            ('{for}', 'vm:atGeoPoly', '{row[area].as_text}'),
            ('{for}', 'vm:withGeoPath', '{row[dataLocation].as_text}'),
            ('{for}', defaults['up'], prefix + '{row[pqcontents].as_slug}'),
        ])
    else:
        transform['queries'] = {
            'for': (
                'select ?maybe_for where {'
                '  { ?maybe_for ?p ?o } union { ?s ?p ?maybe_for }'
                '} limit 1'),
        }
        if prefix == 'vm:_':
            transform['lets']['maybe_for'] = prefix + 'thing_{row[qcontents].as_slug}'
        else:
            transform['lets']['maybe_for'] = prefix + '_{row[qcontents].as_slug}'
        transform['triples'].extend([
            ('{maybe_for}', 'vm:atGeoPoly', '{row[area].as_text}'),
            ('{maybe_for}', 'vm:withGeoPath', '{row[dataLocation].as_text}'),
        ])

    return transform


def remaining_layers(layers, only_layers):
    """Turn only_layer into the set of layers to omit instead."""
    return () if not only_layers else set(
        layer['name'] for layer in layers
        if not layer['name'].startswith(tuple(only_layers))
    )


def _rewrite_class_arg(arg):
    for op in ('==', '*='):
        key, operator, value = arg.partition(op)
        if operator:
            return (operator, key, value)
    raise argparse.ArgumentTypeError(
        'arguments must be key-value terms separated by a *= or == operator, e.g. category==vm:Category or vm:*=vm:HE/')


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoding', default='utf-8')
    parser.add_argument('--name-layer')
    parser.add_argument('--output', metavar='PATH')
    parser.add_argument('--individual-default-class', default='owl:Thing')
    parser.add_argument('--individual-prefix', default='vm:_')
    parser.add_argument(
        '-s', '--source', action='store_true',
        help='Use labels json as source data for model, e.g. creating Activity objects from Activity labels.')
    parser.add_argument(
        '--up-predicate', default='vm:broader',
        help='IRI for predicate to use for up relationships from source.')
    parser.add_argument(
        '--rewrite-class', action='append', type=_rewrite_class_arg,
        default=[('*=', 'https:', 'http:'), ('==', 'activity', 'vm:HE/Activity'), ('==', 'category', 'vm:Category')],
        help='A <key><operator><value> argument describing how the class should be rewritten, for example vm:*=vm:HE/ .'
        ' Valid operators are == , which will fully replace any whole class string exactly matching the key with the value,'
        ' and *= , which will replace any class prefix matching the key with the value while retaining the rest of the'
        ' string.')
    parser.add_argument('-d', '--lens-detail', action='store_true')
    parser.add_argument('input')

    trackergroup = parser.add_mutually_exclusive_group()
    trackergroup.add_argument(
        '-q', '--quiet', action='store_true',
        help='Do not print tree to console as it is processed.'
    )
    trackergroup.add_argument(
        '--export-tabular', metavar='PATH',
        help='Also record all labels in tabular format to another file.')

    layergroup = parser.add_mutually_exclusive_group()
    layergroup.add_argument('--omit-layer', action='append')
    layergroup.add_argument('--only-layer', action='append', help='Only include labels from this layer.')

    args = parser.parse_args(argv[1:])

    with open(args.input, encoding=args.encoding) as f:
        data = json.load(f)

    omit_layers = args.omit_layer or remaining_layers(
        data['layers'], args.only_layer)

    with build_tracker(args) as tracking_fn:
        out_data = pprint.pformat(to_transform(
            data, args.source, args.name_layer, omit_layers,
            args.individual_prefix, {
                'class': args.individual_default_class,
                'rewrite_class': args.rewrite_class,
                'up': args.up_predicate,
                'lens_detail': args.lens_detail,
                'tracking': tracking_fn,
            },
        ))

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f_out:
            f_out.write(out_data)
    else:
        print(out_data)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
