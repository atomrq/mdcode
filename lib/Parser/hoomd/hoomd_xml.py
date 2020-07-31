from numpy import array, loadtxt
import numpy
from io import StringIO
from xml.etree import cElementTree  # pypy will be a bit slower than python
import warnings
from DataStructure.DtypeDict import dtypeDict

try:
    from iopro import loadtxt
except ImportError:
    warnings.warn("No module iopro, I can't accelerate while files are large.")


class hoomd_xml(object):
    @staticmethod
    def _get_attrib(dd):
        dt = eval('[' + ','.join(["('%s', int)" % key for key in dd.keys()]) + ']')
        values = [tuple(dd.values())]
        return array(values, dtype=dt)

    def __init__(self, filename, needed=[]):
        tree = cElementTree.ElementTree(file=filename)
        root = tree.getroot()
        configuration = root[0]
        self.configure = self._get_attrib(configuration.attrib)
        self.nodes = {}
        for e in configuration:
            if e.tag == 'box':
                self.cbox = self._get_attrib(e.attrib)
                self.box = numpy.array([self.cbox['lx'], self.cbox['ly'], self.cbox['lz']]).reshape((3,)).astype(numpy.float64)
                continue
            if (len(needed) != 0) and (not e.tag in needed):
                continue
            try:
                # print(e.tag)
                dt = dtypeDict[e.tag]
                self.nodes[e.tag] = loadtxt(StringIO(e.text), dtype=dt)
            except KeyError:
                self.nodes[e.tag] = loadtxt(StringIO(e.text))
