/*
 * LineTriage - ESP32-CAM Firmware
 * Board: AI Thinker ESP32-CAM
 * Upload: Connect GPIO0 to GND, upload, disconnect, reset
 *
 * Endpoints:
 *   GET /capture              → JPEG image
 *   GET /motor?cmd=start      → Start conveyor
 *   GET /motor?cmd=stop       → Stop conveyor
 *   GET /motor?cmd=speed&value=200
 *   GET /reject               → Actuate servo
 *   GET /status               → JSON status
 */

#include <WiFi.h>
#include <WebServer.h>
#include <esp_camera.h>
#include <ESP32Servo.h>

// ============================================================
// WIFI
// ============================================================
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// ============================================================
// CAMERA PINS (AI-Thinker)
// ============================================================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ============================================================
// MOTOR + SERVO PINS
// ============================================================
#define MOTOR_IN1      12
#define MOTOR_IN2      13
#define MOTOR_ENA      14
#define SERVO_PIN      15

// ============================================================
// GLOBALS
// ============================================================
WebServer server(80);
Servo rejectionServo;
bool isRunning = false;
int motorSpeed = 200;

// ============================================================
// /capture
// ============================================================
void handleCapture() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        server.send(500, "text/plain", "Capture failed");
        return;
    }
    Serial.printf("Image: %dx%d, %d bytes\n", fb->width, fb->height, fb->len);
    server.setContentLength(fb->len);
    server.send(200, "image/jpeg");
    WiFiClient client = server.client();
    client.write(fb->buf, fb->len);
    esp_camera_fb_return(fb);
}

// ============================================================
// /motor
// ============================================================
void handleMotor() {
    String cmd = server.arg("cmd");
    String response = "";
    if (cmd == "start") {
        digitalWrite(MOTOR_IN1, HIGH);
        digitalWrite(MOTOR_IN2, LOW);
        analogWrite(MOTOR_ENA, motorSpeed);
        isRunning = true;
        response = "Motor started at " + String(motorSpeed);
    } else if (cmd == "stop") {
        analogWrite(MOTOR_ENA, 0);
        digitalWrite(MOTOR_IN1, LOW);
        digitalWrite(MOTOR_IN2, LOW);
        isRunning = false;
        response = "Motor stopped";
    } else if (cmd == "speed") {
        int value = server.arg("value").toInt();
        if (value >= 0 && value <= 255) {
            motorSpeed = value;
            if (isRunning) analogWrite(MOTOR_ENA, motorSpeed);
            response = "Speed set to " + String(motorSpeed);
        } else {
            response = "Invalid speed";
        }
    }
    server.send(200, "text/plain", response);
}

// ============================================================
// /reject
// ============================================================
void handleReject() {
    Serial.println("REJECT");
    rejectionServo.write(90);
    delay(500);
    rejectionServo.write(0);
    delay(500);
    server.send(200, "text/plain", "Rejected");
}

// ============================================================
// /status
// ============================================================
void handleStatus() {
    String s = "{";
    s += "\"running\":" + String(isRunning ? "true" : "false") + ",";
    s += "\"speed\":" + String(motorSpeed) + "}";
    server.send(200, "application/json", s);
}

// ============================================================
// SETUP
// ============================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\nLineTriage ESP32-CAM Starting...");

    pinMode(MOTOR_IN1, OUTPUT);
    pinMode(MOTOR_IN2, OUTPUT);
    pinMode(MOTOR_ENA, OUTPUT);
    analogWrite(MOTOR_ENA, 0);

    rejectionServo.attach(SERVO_PIN);
    rejectionServo.write(0);

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return;
    }
    Serial.println("Camera initialized");

    WiFi.begin(ssid, password);
    Serial.print("Connecting WiFi");
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi Connected");
        Serial.print("IP: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\nWiFi failed");
    }

    server.on("/capture", HTTP_GET, handleCapture);
    server.on("/motor", HTTP_GET, handleMotor);
    server.on("/reject", HTTP_GET, handleReject);
    server.on("/status", HTTP_GET, handleStatus);
    server.begin();
    Serial.println("HTTP server started on port 80");
}

void loop() {
    server.handleClient();
    delay(10);
}