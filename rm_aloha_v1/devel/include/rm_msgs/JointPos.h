// Generated by gencpp from file rm_msgs/JointPos.msg
// DO NOT EDIT!


#ifndef RM_MSGS_MESSAGE_JOINTPOS_H
#define RM_MSGS_MESSAGE_JOINTPOS_H


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
struct JointPos_
{
  typedef JointPos_<ContainerAllocator> Type;

  JointPos_()
    : joint()
    , expand(0.0)  {
    }
  JointPos_(const ContainerAllocator& _alloc)
    : joint(_alloc)
    , expand(0.0)  {
  (void)_alloc;
    }



   typedef std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> _joint_type;
  _joint_type joint;

   typedef float _expand_type;
  _expand_type expand;





  typedef boost::shared_ptr< ::rm_msgs::JointPos_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::rm_msgs::JointPos_<ContainerAllocator> const> ConstPtr;

}; // struct JointPos_

typedef ::rm_msgs::JointPos_<std::allocator<void> > JointPos;

typedef boost::shared_ptr< ::rm_msgs::JointPos > JointPosPtr;
typedef boost::shared_ptr< ::rm_msgs::JointPos const> JointPosConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::rm_msgs::JointPos_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::rm_msgs::JointPos_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::rm_msgs::JointPos_<ContainerAllocator1> & lhs, const ::rm_msgs::JointPos_<ContainerAllocator2> & rhs)
{
  return lhs.joint == rhs.joint &&
    lhs.expand == rhs.expand;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::rm_msgs::JointPos_<ContainerAllocator1> & lhs, const ::rm_msgs::JointPos_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace rm_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::rm_msgs::JointPos_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rm_msgs::JointPos_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rm_msgs::JointPos_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rm_msgs::JointPos_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rm_msgs::JointPos_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rm_msgs::JointPos_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::rm_msgs::JointPos_<ContainerAllocator> >
{
  static const char* value()
  {
    return "278dfe354f7a05e095637c9789d0fcfe";
  }

  static const char* value(const ::rm_msgs::JointPos_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x278dfe354f7a05e0ULL;
  static const uint64_t static_value2 = 0x95637c9789d0fcfeULL;
};

template<class ContainerAllocator>
struct DataType< ::rm_msgs::JointPos_<ContainerAllocator> >
{
  static const char* value()
  {
    return "rm_msgs/JointPos";
  }

  static const char* value(const ::rm_msgs::JointPos_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::rm_msgs::JointPos_<ContainerAllocator> >
{
  static const char* value()
  {
    return "#control arm joints without planning\n"
"\n"
"float32[] joint\n"
"float32    expand\n"
;
  }

  static const char* value(const ::rm_msgs::JointPos_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::rm_msgs::JointPos_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.joint);
      stream.next(m.expand);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct JointPos_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::rm_msgs::JointPos_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::rm_msgs::JointPos_<ContainerAllocator>& v)
  {
    s << indent << "joint[]" << std::endl;
    for (size_t i = 0; i < v.joint.size(); ++i)
    {
      s << indent << "  joint[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.joint[i]);
    }
    s << indent << "expand: ";
    Printer<float>::stream(s, indent + "  ", v.expand);
  }
};

} // namespace message_operations
} // namespace ros

#endif // RM_MSGS_MESSAGE_JOINTPOS_H
