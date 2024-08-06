;; Auto-generated. Do not edit!


(when (boundp 'rm_msgs::MoveJ)
  (if (not (find-package "RM_MSGS"))
    (make-package "RM_MSGS"))
  (shadow 'MoveJ (find-package "RM_MSGS")))
(unless (find-package "RM_MSGS::MOVEJ")
  (make-package "RM_MSGS::MOVEJ"))

(in-package "ROS")
;;//! \htmlinclude MoveJ.msg.html


(defclass rm_msgs::MoveJ
  :super ros::object
  :slots (_joint _speed ))

(defmethod rm_msgs::MoveJ
  (:init
   (&key
    ((:joint __joint) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:speed __speed) 0.0)
    )
   (send-super :init)
   (setq _joint __joint)
   (setq _speed (float __speed))
   self)
  (:joint
   (&optional __joint)
   (if __joint (setq _joint __joint)) _joint)
  (:speed
   (&optional __speed)
   (if __speed (setq _speed __speed)) _speed)
  (:serialization-length
   ()
   (+
    ;; float32[] _joint
    (* 4    (length _joint)) 4
    ;; float32 _speed
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _joint
     (write-long (length _joint) s)
     (dotimes (i (length _joint))
       (sys::poke (elt _joint i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32 _speed
       (sys::poke _speed (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _joint
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _joint (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _joint i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32 _speed
     (setq _speed (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get rm_msgs::MoveJ :md5sum-) "4431244c9810cafe3ab938cac63e0037")
(setf (get rm_msgs::MoveJ :datatype-) "rm_msgs/MoveJ")
(setf (get rm_msgs::MoveJ :definition-)
      "float32[] joint
float32 speed

")



(provide :rm_msgs/MoveJ "4431244c9810cafe3ab938cac63e0037")


