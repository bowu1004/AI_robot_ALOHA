// Generated by gencpp from file rm_msgs/Force_Position_Move_Joint.msg
// DO NOT EDIT!


#ifndef RM_MSGS_MESSAGE_FORCE_POSITION_MOVE_JOINT_H
#define RM_MSGS_MESSAGE_FORCE_POSITION_MOVE_JOINT_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace rm_msgs
{
template <class ContainerAllocator>
struct Force_Position_Move_Joint_
{
  typedef Force_Position_Move_Joint_<ContainerAllocator> Type;

  Force_Position_Move_Joint_()
    : joint()
    , sensor(0)
    , mode(0)
    , dir(0)
    , force(0)
    , dof(0)  {
    }
  Force_Position_Move_Joint_(const ContainerAllocator& _alloc)
    : joint(_alloc)
    , sensor(0)
    , mode(0)
    , dir(0)
    , force(0)
    , dof(0)  {
  (void)_alloc;
    }



   typedef std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> _joint_type;
  _joint_type joint;

   typedef uint8_t _sensor_type;
  _sensor_type sensor;

   typedef uint8_t _mode_type;
  _mode_type mode;

   typedef uint8_t _dir_type;
  _dir_type dir;

   typedef int16_t _force_type;
  _force_type force;

   typedef uint8_t _dof_type;
  _dof_type dof;





  typedef boost::shared_ptr< ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator> const> ConstPtr;

}; // struct Force_Position_Move_Joint_

typedef ::rm_msgs::Force_Position_Move_Joint_<std::allocator<void> > Force_Position_Move_Joint;

typedef boost::shared_ptr< ::rm_msgs::Force_Position_Move_Joint > Force_Position_Move_JointPtr;
typedef boost::shared_ptr< ::rm_msgs::Force_Position_Move_Joint const> Force_Position_Move_JointConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator1> & lhs, const ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator2> & rhs)
{
  return lhs.joint == rhs.joint &&
    lhs.sensor == rhs.sensor &&
    lhs.mode == rhs.mode &&
    lhs.dir == rhs.dir &&
    lhs.force == rhs.force &&
    lhs.dof == rhs.dof;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator1> & lhs, const ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace rm_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator> >
{
  static const char* value()
  {
    return "77bde1aba3500cfee05e409713ffba41";
  }

  static const char* value(const ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x77bde1aba3500cfeULL;
  static const uint64_t static_value2 = 0xe05e409713ffba41ULL;
};

template<class ContainerAllocator>
struct DataType< ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator> >
{
  static const char* value()
  {
    return "rm_msgs/Force_Position_Move_Joint";
  }

  static const char* value(const ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float32[] joint\n"
"uint8 sensor\n"
"uint8 mode\n"
"uint8 dir\n"
"int16 force\n"
"uint8 dof\n"
;
  }

  static const char* value(const ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.joint);
      stream.next(m.sensor);
      stream.next(m.mode);
      stream.next(m.dir);
      stream.next(m.force);
      stream.next(m.dof);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Force_Position_Move_Joint_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::rm_msgs::Force_Position_Move_Joint_<ContainerAllocator>& v)
  {
    s << indent << "joint[]" << std::endl;
    for (size_t i = 0; i < v.joint.size(); ++i)
    {
      s << indent << "  joint[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.joint[i]);
    }
    s << indent << "sensor: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.sensor);
    s << indent << "mode: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.mode);
    s << indent << "dir: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.dir);
    s << indent << "force: ";
    Printer<int16_t>::stream(s, indent + "  ", v.force);
    s << indent << "dof: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.dof);
  }
};

} // namespace message_operations
} // namespace ros

#endif // RM_MSGS_MESSAGE_FORCE_POSITION_MOVE_JOINT_H
