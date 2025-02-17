// Generated by gencpp from file rm_msgs/Joint_Max_Speed.msg
// DO NOT EDIT!


#ifndef RM_MSGS_MESSAGE_JOINT_MAX_SPEED_H
#define RM_MSGS_MESSAGE_JOINT_MAX_SPEED_H


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
struct Joint_Max_Speed_
{
  typedef Joint_Max_Speed_<ContainerAllocator> Type;

  Joint_Max_Speed_()
    : joint_num(0)
    , joint_max_speed(0.0)  {
    }
  Joint_Max_Speed_(const ContainerAllocator& _alloc)
    : joint_num(0)
    , joint_max_speed(0.0)  {
  (void)_alloc;
    }



   typedef uint8_t _joint_num_type;
  _joint_num_type joint_num;

   typedef float _joint_max_speed_type;
  _joint_max_speed_type joint_max_speed;





  typedef boost::shared_ptr< ::rm_msgs::Joint_Max_Speed_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::rm_msgs::Joint_Max_Speed_<ContainerAllocator> const> ConstPtr;

}; // struct Joint_Max_Speed_

typedef ::rm_msgs::Joint_Max_Speed_<std::allocator<void> > Joint_Max_Speed;

typedef boost::shared_ptr< ::rm_msgs::Joint_Max_Speed > Joint_Max_SpeedPtr;
typedef boost::shared_ptr< ::rm_msgs::Joint_Max_Speed const> Joint_Max_SpeedConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::rm_msgs::Joint_Max_Speed_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::rm_msgs::Joint_Max_Speed_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::rm_msgs::Joint_Max_Speed_<ContainerAllocator1> & lhs, const ::rm_msgs::Joint_Max_Speed_<ContainerAllocator2> & rhs)
{
  return lhs.joint_num == rhs.joint_num &&
    lhs.joint_max_speed == rhs.joint_max_speed;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::rm_msgs::Joint_Max_Speed_<ContainerAllocator1> & lhs, const ::rm_msgs::Joint_Max_Speed_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace rm_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::rm_msgs::Joint_Max_Speed_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rm_msgs::Joint_Max_Speed_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rm_msgs::Joint_Max_Speed_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rm_msgs::Joint_Max_Speed_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rm_msgs::Joint_Max_Speed_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rm_msgs::Joint_Max_Speed_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::rm_msgs::Joint_Max_Speed_<ContainerAllocator> >
{
  static const char* value()
  {
    return "9a15b693ccbb220eba8aa0b693b24585";
  }

  static const char* value(const ::rm_msgs::Joint_Max_Speed_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x9a15b693ccbb220eULL;
  static const uint64_t static_value2 = 0xba8aa0b693b24585ULL;
};

template<class ContainerAllocator>
struct DataType< ::rm_msgs::Joint_Max_Speed_<ContainerAllocator> >
{
  static const char* value()
  {
    return "rm_msgs/Joint_Max_Speed";
  }

  static const char* value(const ::rm_msgs::Joint_Max_Speed_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::rm_msgs::Joint_Max_Speed_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uint8 joint_num\n"
"float32 joint_max_speed\n"
;
  }

  static const char* value(const ::rm_msgs::Joint_Max_Speed_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::rm_msgs::Joint_Max_Speed_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.joint_num);
      stream.next(m.joint_max_speed);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Joint_Max_Speed_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::rm_msgs::Joint_Max_Speed_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::rm_msgs::Joint_Max_Speed_<ContainerAllocator>& v)
  {
    s << indent << "joint_num: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.joint_num);
    s << indent << "joint_max_speed: ";
    Printer<float>::stream(s, indent + "  ", v.joint_max_speed);
  }
};

} // namespace message_operations
} // namespace ros

#endif // RM_MSGS_MESSAGE_JOINT_MAX_SPEED_H
