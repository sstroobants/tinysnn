struct __attribute__((__packed__)) serial_pid_out {
    //roll commands
    float roll_p; //roll p
    float roll_i; //roll i
    float roll_d; //roll d
    //pitch commands
    float pitch_p; //pitch p
    float pitch_i; //pitch i
    float pitch_d; //pitch d
    //yaw commands
    float yaw_p; //yaw p
    float yaw_i; //yaw i
    float yaw_d; //yaw d
     //Rolling message out
    // float rolling_msg_out;
    // uint8_t rolling_msg_out_id;
    //CHECKSUM
    uint8_t checksum_out;
};