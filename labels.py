#!/usr/bin/env python3

"""Turn json ai semantic export into a transform."""

import argparse
import base64
import hashlib
import itertools
import json
import pprint
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
    return '#' + string.lstrip('#')


def deriveClass(url):
    if url == 'activity':
        return 'vm:HE/Activity'
    if url == 'category':
        return 'vm:Category'
    return url.replace('https:', 'http:')


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
    return {k.strip(): v.strip().strip('"') for k, _, v in entries}


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
    lines = string.strip().replace('\u0003', ' ').split('\r')
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
    return [[alist[i-1], alist[i]] for i in range(1, len(alist), 2)]


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
            print ('DUPE', contents.replace('\r', ' '), file=sys.stderr)
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
            classFor=deriveClass(note.pop('class', parent.get('class', defaults['class']))),
            note=note,
            **label['name'],
        )


def iter_labels(layers, scaler, defaults):
    counter = itertools.count()
    stack = list(layers)
    stack.reverse()
    while stack:
        o = stack.pop()
        stack.extend(map_labels(reversed(o['children']), o, counter, defaults))
        print(
            '{indent}{name}'.format(
                indent=o['depth'] * '  ',
                name=o.get('name') or o['text'].replace('\n', ' ')),
            file=sys.stderr)
        if o.get('type'):
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
                background=fixColor(o['note'].get('background')),
                color=deriveColor(o['characterColour']),
                text=o['text'],
                fontFamily=o['typeface'],
                fontSize=scaler.distance(o['fontSize']),
                # TODO: matrix, opacity
            )
            row['display'] = json_str(display)
        # Experiment with including point even for hidden labels
        row['geoPoint'] = json_str(scaler.latlng(o['centrePoint']))
        yield row


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
        #assert scale_x == scale_y
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
            'layer': '{row[layer].as_text}',
            'content': '{row[qcontents].as_text}',
        },
        'triples': [
            ('{iri}', 'rdf:type', 'vm:Label'),
            ('{iri}', 'vm:atGeoPoint', '{row[geoPoint].as_text}'),
            ('{iri}', 'vm:display', '{row[display].as_text}'),
            ('{iri}', 'vm:forThing', '{for}'),
            ('{iri}', 'vm:forLayer', '{layer}'),
        ],
    }

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
        transform['lets']['maybe_for'] = 'vm:_thing_{row[qcontents].as_slug}'
        transform['triples'].extend([
            ('{maybe_for}', 'vm:atGeoPoly', '{row[area].as_text}'),
            ('{maybe_for}', 'vm:withGeoPath', '{row[dataLocation].as_text}'),
        ])

    return transform


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoding', default='utf-8')
    parser.add_argument('--name-layer')
    parser.add_argument('--omit-layer', action='append')
    parser.add_argument('--output')
    parser.add_argument('--individual-default-class', default='owl:Thing')
    parser.add_argument('--individual-prefix', default='vm:_')
    parser.add_argument('-s', '--source', action='store_true',
        help='Use labels json as source data for model, e.g. creating Activity objects from Activity labels.')
    parser.add_argument('--up-predicate', default='vm:broader',
        help='IRI for predicate to use for up relationships from source.')
    parser.add_argument('input')
    args = parser.parse_args(argv[1:])

    with open(args.input, encoding=args.encoding) as f:
        data = json.load(f)

    out_data = pprint.pformat(to_transform(
        data, args.source, args.name_layer, args.omit_layer or (),
        args.individual_prefix, {
            'class': args.individual_default_class,
            'up': args.up_predicate,
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
