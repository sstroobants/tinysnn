#include <Arduino.h>

#include "msgs/pidInput.h"
#include "msgs/pidOutput.h"

extern "C"{
    #include "pid.h"
    #include "functional.h"
    #include "param/test_pid_conf.h"
}

// -------------------------- COMMUNICATION DEFINED VARIABLES-----------------------------
#define COMMUNICATION_SERIAL Serial2
#define COMMUNICATION_SERIAL_BAUD 460800

byte START_BYTE_SERIAL_CF=0x9A;
elapsedMicros last_time_write_to_cf = 0;
int ack_comm_TX_cf = 0; 

struct serial_pid_in myserial_pid_in;
uint8_t serial_cf_msg_buf_in[ 2*sizeof(struct serial_pid_in) ] = {0};
uint16_t serial_cf_buf_in_cnt = 0;
int serial_cf_received_packets = 0;
int serial_cf_missed_packets_in = 0;

volatile struct serial_pid_out myserial_pid_out;
volatile float extra_data_out[255]__attribute__((aligned));

bool sending;
bool receiving = true;

// -------------------------- DEBUG DEFINED VARIABLES-----------------------------
#define DEBUG_serial Serial
#define DEBUG_serial_baud 115200

// -------------------------- PID DEFINED VARIABLES-------------------------------
elapsedMicros timer_count_main = 0;
PID rollPid;
PID pitchPid;
PID yawPid;

float errorRoll;
float stateRoll;
float errorPitch;
float statePitch;
float errorYaw;
float stateYaw;

void setup(void) 
{
    //////////////////SETUP DEBUGGING USB
    DEBUG_serial.begin(DEBUG_serial_baud);


    //////////////////Initialize PID
    rollPid = build_pid(2, 80, 1);
    pitchPid = build_pid(2, 80, 1);
    yawPid = build_pid(2, 80, 1);
    init_pid(&rollPid);
    init_pid(&pitchPid);
    init_pid(&yawPid);

    // Load network parameters from header file
    load_pid_from_header(&pitchPid, &conf);
    load_pid_from_header(&rollPid, &conf);
    load_pid_from_header(&yawPid, &conf);

    reset_pid(&rollPid);
    reset_pid(&pitchPid);
    reset_pid(&yawPid);

    //////////////////SETUP CONNECTION WITH CRAZYFLIE
    COMMUNICATION_SERIAL.begin(COMMUNICATION_SERIAL_BAUD);
    DEBUG_serial.write("Finished setup\n");
}

///////////////////////////////////////////////////////////LOOP///////////////////
void loop(void) 
{
    if (receiving) {
        // DEBUG_serial.write("receiving...");
        receiveCrazyflie();
    } else if (sending) {
        // Timer for debugging
        if (timer_count_main > 1000000) {
          DEBUG_serial.printf("Received %i packets over last second\n", serial_cf_received_packets);
          serial_cf_received_packets = 0;
          timer_count_main = 0;
        }
        // Set input to network from CF
        // DEBUG_serial.write("Setting input message\n");
        setInputMessage();

        // Forward network
        // DEBUG_serial.write("PID step\n");
        if (myserial_pid_in.thrust == 0.0f) {
          reset_pid(&rollPid);
          reset_pid(&pitchPid);
          reset_pid(&yawPid);
        }
        forward_pid(&rollPid);
        forward_pid(&pitchPid);
        // forward_pid(&yawPid);

        // Store output message to be sent back to CF
        setOutputMessage();

        // Send message via UART to CF
        sendCrazyflie();
    }
}


///////////////////////////////////////////////USER DEFINED FCN///////////////////
void serialParseMessageIn(void)
{
  //Copy received buffer to structure
  memmove(&myserial_pid_in,&serial_cf_msg_buf_in[1],sizeof(struct serial_pid_in)-1);
  // DEBUG_serial.write("Correct message received and storing\n");
//   DEBUG_serial.write("Stored pitch is %i\n", myserial_pid_in.pitch);
}

void setInputMessage(void)
{
    errorRoll = (myserial_pid_in.roll - myserial_pid_in.roll_t) * 0.005;
    // float state = (- myserial_pid_in.pitch - myserial_pid_in.pitch_t) * 0.01;
    stateRoll = myserial_pid_in.roll * 0.005;
    errorPitch = (- myserial_pid_in.pitch - myserial_pid_in.pitch_t) * 0.005;
    // float state = (- myserial_pid_in.pitch - myserial_pid_in.pitch_t) * 0.01;
    statePitch = - myserial_pid_in.pitch * 0.005;
    
    set_pid_input(&rollPid, errorRoll, stateRoll);
    set_pid_input(&pitchPid, errorPitch, statePitch);
    set_pid_input(&yawPid, errorPitch, statePitch);
}

void setOutputMessage(void)
{
    myserial_pid_out.roll_p = rollPid.out[0] * -12000;
    myserial_pid_out.roll_i = rollPid.out[1] * -10000;
    myserial_pid_out.roll_d = rollPid.out[2] * -10000;
    // myserial_pid_out.roll_p = rollPid.out[0] * -20000;
    // myserial_pid_out.roll_i = rollPid.out[1] * -20000;
    // myserial_pid_out.roll_d = rollPid.out[2] * -20000;
    myserial_pid_out.pitch_p = pitchPid.out[0] * -12000;
    myserial_pid_out.pitch_i = pitchPid.out[1] * -10000;
    myserial_pid_out.pitch_d = pitchPid.out[2] * -10000;
    // myserial_pid_out.pitch_p = pitchPid.out[0] * -20000;
    // myserial_pid_out.pitch_i = pitchPid.out[1] * -20000;
    // myserial_pid_out.pitch_d = pitchPid.out[2] * -20000;
    // myserial_pid_out.pitch_d = pid.out[2] * 0;

    myserial_pid_out.yaw_p = yawPid.out[0] * -10000;
    // myserial_pid_out.pitch_i = pid.out[1] * -20000;
    myserial_pid_out.yaw_i = yawPid.out[1] * -500;
    myserial_pid_out.yaw_d = yawPid.out[2] * -12000;
}

void sendCrazyflie(void)
{
    //SENDING PACKET

    //Calculate checksum for outbound packet: 
    uint8_t *buf_send = (uint8_t *)&myserial_pid_out;
    myserial_pid_out.checksum_out = 0;
    for(uint16_t i = 0; i < sizeof(struct serial_pid_out) - 1; i++){
        myserial_pid_out.checksum_out += buf_send [i];
    }
    
    //Send out packet to buffer:
    noInterrupts();
    COMMUNICATION_SERIAL.write(START_BYTE_SERIAL_CF);
    COMMUNICATION_SERIAL.write(buf_send,sizeof(struct serial_pid_out));
    interrupts();
    
    last_time_write_to_cf = 0;

    sending = false;
    receiving = true;
}


void receiveCrazyflie(void)
{
  //RECEIVING PACKET
  //Collect packets on the buffer if available:
    while(COMMUNICATION_SERIAL.available()) {
        // DEBUG_serial.write("trying to read...\n");
        uint8_t serial_cf_byte_in;
        serial_cf_byte_in = COMMUNICATION_SERIAL.read();
        if ((serial_cf_byte_in == START_BYTE_SERIAL_CF) || (serial_cf_buf_in_cnt > 0)) {
            serial_cf_msg_buf_in[serial_cf_buf_in_cnt] = serial_cf_byte_in;
            serial_cf_buf_in_cnt++;
        }
        if (serial_cf_buf_in_cnt > sizeof(struct serial_pid_in)  ) {
            serial_cf_buf_in_cnt = 0;
            uint8_t checksum_in_local = 0;
            for(uint16_t i = 1; i < sizeof(struct serial_pid_in) ; i++){
                checksum_in_local += serial_cf_msg_buf_in[i];
            }
            if(checksum_in_local == serial_cf_msg_buf_in[sizeof(struct serial_pid_in)]){
                serialParseMessageIn();
                serial_cf_received_packets++;
                
            }
            else {
                serial_cf_missed_packets_in++;           
                DEBUG_serial.write("Incorrect message\n");
            }
            receiving = false;
            sending = true;
        }
    }
}

