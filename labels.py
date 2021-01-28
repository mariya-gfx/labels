#!/usr/bin/env python3

"""Turn json ai semantic export into a transform."""

import argparse
import json
import pprint
import sys


def deriveColour(col):
    """Create CSS color from i9r json."""
    # TODO: Handle non-rgb?
    return 'rgb({red},{green},{blue})'.format(**col)


def deriveIdent(path):
    return '-'.join(path.split('/')[1:]).strip()


def deriveQcon(path):
    parts = path.rsplit('/', 2)[1:]
    if len(parts) == 1:
        return parts[0].strip()
    return ' '.join([parts[0].strip().rstrip('s'), parts[1].strip()])


def deriveText(string):
    return '\n'.join(line.strip() for line in string.split('\r'))


def map_layers(layers):
    for layer in layers:
        if not layer['name'].startswith(('~', '-', '-~')):
            continue
        yield dict(depth=0, uid=layer['name'], **layer)


def map_labels(labels, parent):
    for label in labels:
        if label['name']['contents'] == parent.get('contents'):
            continue  # Skip duplicated immediate child labels
        yield dict(
            depth=parent['depth'] + 1,
            children=label.get('children', ()),
            parent=parent,
            path=label['path'],
            uid=label['uID'],
            **label['name'],
        )


def iter_labels(layers, scaler):
    stack = list(layers)
    stack.reverse()
    while stack:
        o = stack.pop()
        stack.extend(map_labels(reversed(o['children']), o))
        print(
            '{indent}{name}'.format(
                indent=o['depth'] * '  ',
                name=o.get('name') or o.get('contents')),
            file=sys.stderr)
        if o.get('type') or o.get('hidden'):
            continue
        yield dict(
            bounds=repr(scaler.latlng_bounds(o['bounds'])),
            boxFillColour=deriveColour(o['boxFillColour']),
            centrePoint=repr(scaler.latlng(o['centrePoint'])),
            characterColour=deriveColour(o['characterColour']),
            fontSize=repr(scaler.distance(o['fontSize'])),
            contents=deriveText(o['contents']),
            ident=deriveIdent(o['path']),
            qcontents=deriveQcon(o['path']),
            typeface=o['typeface'],
            # TODO: matrix, opacity, area
        )


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
        assert scale_x == scale_y
        return cls(x1, y1, scale_x)

    @classmethod
    def from_artboard(cls, ab):
        return cls.from_rect(ab['left'], ab['top'], ab['right'], ab['bottom'])

    def _to(self, n):
        return round(n * self.scale, self.ROUND)

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


def to_transform(data):
    scaler = Scaler.from_artboard(data['artboard'])
    rows = list(iter_labels(map_layers(data['layers']), scaler))
    # TODO: merge labels across views
    return {
        'data': rows,
        'lets': {
            'iri': 'vm:label-{row[ident].as_slug}',
            'name': '{row[contents].as_text}',
            'nname': '{row[qcontents].as_text}',
        },
        'queries': {
            'for': '''
                select ?t where {
                    ?t vm:name ?o .
                    filter (?o in (?name, ?nname))
                }
            ''',
        },
        'triples': [
            ('{iri}', 'rdf:type', 'vm:Label'),
            ('{iri}', 'vm:fontSize', '{row[fontSize]}'),
            ('{iri}', 'vm:fontFamily', '{row[typeface]}'),
            ('{iri}', 'vm:description', '{name}'),
            ('{iri}', 'vm:atGeoPoint', '{row[centrePoint]}'),
            ('{iri}', 'vm:fontColor', '{row[characterColour]}'),
            ('{iri}', 'vm:boxBounds', '{row[bounds]}'),
            ('{iri}', 'vm:boxColor', '{row[boxFillColour]}'),
            ('{iri}', 'vm:forThing', '{for}'),
        ],
    }


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoding', default='utf-8')
    parser.add_argument('input')
    args = parser.parse_args(argv[1:])

    with open(args.input, encoding=args.encoding) as f:
        data = json.load(f)

    pprint.pprint(to_transform(data))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
