# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from rm_msgs/Set_Realtime_Push.msg. Do not edit."""
import codecs
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct


class Set_Realtime_Push(genpy.Message):
  _md5sum = "9a0e0df44121dc8d27005a2fbd40ac91"
  _type = "rm_msgs/Set_Realtime_Push"
  _has_header = False  # flag to mark the presence of a Header object
  _full_text = """uint16 cycle
uint16 port
uint16 force_coordinate
string ip"""
  __slots__ = ['cycle','port','force_coordinate','ip']
  _slot_types = ['uint16','uint16','uint16','string']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       cycle,port,force_coordinate,ip

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(Set_Realtime_Push, self).__init__(*args, **kwds)
      # message fields cannot be None, assign default values for those that are
      if self.cycle is None:
        self.cycle = 0
      if self.port is None:
        self.port = 0
      if self.force_coordinate is None:
        self.force_coordinate = 0
      if self.ip is None:
        self.ip = ''
    else:
      self.cycle = 0
      self.port = 0
      self.force_coordinate = 0
      self.ip = ''

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self
      buff.write(_get_struct_3H().pack(_x.cycle, _x.port, _x.force_coordinate))
      _x = self.ip
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      end = 0
      _x = self
      start = end
      end += 6
      (_x.cycle, _x.port, _x.force_coordinate,) = _get_struct_3H().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.ip = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.ip = str[start:end]
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self
      buff.write(_get_struct_3H().pack(_x.cycle, _x.port, _x.force_coordinate))
      _x = self.ip
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      end = 0
      _x = self
      start = end
      end += 6
      (_x.cycle, _x.port, _x.force_coordinate,) = _get_struct_3H().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.ip = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.ip = str[start:end]
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_3H = None
def _get_struct_3H():
    global _struct_3H
    if _struct_3H is None:
        _struct_3H = struct.Struct("<3H")
    return _struct_3H
