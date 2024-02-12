struct __attribute__((__packed__)) serial_control_out {
    //torque commands
    int16_t torque_x; //torque x
    int16_t torque_y; //torque y
    int16_t torque_z; //torque z
    int16_t x_integ;
    int16_t y_integ;
    //CHECKSUM
    uint8_t checksum_out;
};