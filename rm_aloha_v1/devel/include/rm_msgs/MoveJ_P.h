// Generated by gencpp from file rm_msgs/MoveJ_P.msg
// DO NOT EDIT!


#ifndef RM_MSGS_MESSAGE_MOVEJ_P_H
#define RM_MSGS_MESSAGE_MOVEJ_P_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <geometry_msgs/Pose.h>

namespace rm_msgs
{
template <class ContainerAllocator>
struct MoveJ_P_
{
  typedef MoveJ_P_<ContainerAllocator> Type;

  MoveJ_P_()
    : Pose()
    , speed(0.0)  {
    }
  MoveJ_P_(const ContainerAllocator& _alloc)
    : Pose(_alloc)
    , speed(0.0)  {
  (void)_alloc;
    }



   typedef  ::geometry_msgs::Pose_<ContainerAllocator>  _Pose_type;
  _Pose_type Pose;

   typedef float _speed_type;
  _speed_type speed;





  typedef boost::shared_ptr< ::rm_msgs::MoveJ_P_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::rm_msgs::MoveJ_P_<ContainerAllocator> const> ConstPtr;

}; // struct MoveJ_P_

typedef ::rm_msgs::MoveJ_P_<std::allocator<void> > MoveJ_P;

typedef boost::shared_ptr< ::rm_msgs::MoveJ_P > MoveJ_PPtr;
typedef boost::shared_ptr< ::rm_msgs::MoveJ_P const> MoveJ_PConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::rm_msgs::MoveJ_P_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::rm_msgs::MoveJ_P_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::rm_msgs::MoveJ_P_<ContainerAllocator1> & lhs, const ::rm_msgs::MoveJ_P_<ContainerAllocator2> & rhs)
{
  return lhs.Pose == rhs.Pose &&
    lhs.speed == rhs.speed;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::rm_msgs::MoveJ_P_<ContainerAllocator1> & lhs, const ::rm_msgs::MoveJ_P_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace rm_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::rm_msgs::MoveJ_P_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rm_msgs::MoveJ_P_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rm_msgs::MoveJ_P_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rm_msgs::MoveJ_P_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rm_msgs::MoveJ_P_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rm_msgs::MoveJ_P_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::rm_msgs::MoveJ_P_<ContainerAllocator> >
{
  static const char* value()
  {
    return "0bdc59c6a7f930de77636565f2d8ae52";
  }

  static const char* value(const ::rm_msgs::MoveJ_P_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x0bdc59c6a7f930deULL;
  static const uint64_t static_value2 = 0x77636565f2d8ae52ULL;
};

template<class ContainerAllocator>
struct DataType< ::rm_msgs::MoveJ_P_<ContainerAllocator> >
{
  static const char* value()
  {
    return "rm_msgs/MoveJ_P";
  }

  static const char* value(const ::rm_msgs::MoveJ_P_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::rm_msgs::MoveJ_P_<ContainerAllocator> >
{
  static const char* value()
  {
    return "geometry_msgs/Pose Pose\n"
"float32 speed\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Pose\n"
"# A representation of pose in free space, composed of position and orientation. \n"
"Point position\n"
"Quaternion orientation\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Point\n"
"# This contains the position of a point in free space\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Quaternion\n"
"# This represents an orientation in free space in quaternion form.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"float64 w\n"
;
  }

  static const char* value(const ::rm_msgs::MoveJ_P_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::rm_msgs::MoveJ_P_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.Pose);
      stream.next(m.speed);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct MoveJ_P_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::rm_msgs::MoveJ_P_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::rm_msgs::MoveJ_P_<ContainerAllocator>& v)
  {
    s << indent << "Pose: ";
    s << std::endl;
    Printer< ::geometry_msgs::Pose_<ContainerAllocator> >::stream(s, indent + "  ", v.Pose);
    s << indent << "speed: ";
    Printer<float>::stream(s, indent + "  ", v.speed);
  }
};

} // namespace message_operations
} // namespace ros

#endif // RM_MSGS_MESSAGE_MOVEJ_P_H
