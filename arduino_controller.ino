/*
 * Ironing Robot — Arduino Controller
 * ====================================
 * Receives commands from Python over Serial and executes them.
 *
 * Hardware:
 *   - Arduino Uno + CNC Shield R3
 *   - 3x DRV8825 stepper drivers (X, Y axes + spare)
 *   - Stepper motors: NEMA17, 200 steps/rev
 *   - Microstepping: 1/4  →  800 steps/rev
 *   - GT2 belt, 20-tooth pulley  →  40mm/rev
 *   - Steps/mm = 800 / 40 = 20 steps/mm
 *   - Steam relay  : pin 8  (active HIGH)
 *   - Vacuum relay : pin 9  (active HIGH)
 *   - Servo (MG996R): pin 10
 *
 * Protocol:
 *   Python sends:  "MOVE X<mm> Y<mm> F<feed>\n"
 *                  "STEAM_ON\n"  /  "STEAM_OFF\n"
 *                  "VACUUM_ON\n" /  "VACUUM_OFF\n"
 *                  "SERVO_ROTATE\n"
 *                  "HOME\n"
 *   Arduino replies: "OK\n" when the command is complete.
 *
 * Coordinate system:
 *   Home position = (0, 0) = bottom-right corner of garment
 *   X increases going LEFT  (right → left ironing direction)
 *   Y increases going UP    (bottom → top direction)
 *   All positions sent as mm; converted to steps internally.
 */

#include <Servo.h>

// ─────────────────────────────────────────────
// PIN DEFINITIONS
// ─────────────────────────────────────────────

// X-axis stepper (horizontal — left/right)
#define X_STEP_PIN   2
#define X_DIR_PIN    5

// Y-axis stepper (vertical — up/down)
// CNC Shield uses two Y motors wired together for the gantry
#define Y_STEP_PIN   3
#define Y_DIR_PIN    6

// Enable pin (LOW = motors enabled on DRV8825)
#define ENABLE_PIN   8

// Relays (active HIGH — relay board triggers when pin is HIGH)
#define STEAM_PIN    9
#define VACUUM_PIN   10

// Servo
#define SERVO_PIN    11

// ─────────────────────────────────────────────
// MOTION CONFIGURATION
// ─────────────────────────────────────────────
const float STEPS_PER_MM    = 20.0;   // 800 steps/rev ÷ 40mm/rev
const int   STEP_PULSE_US   = 5;      // Stepper pulse width in microseconds (DRV8825 min = 1.9µs)
const float DEFAULT_FEED    = 3000.0; // Default feed rate mm/min
const float MAX_FEED        = 6000.0; // Safety cap on feed rate mm/min

// Servo positions
const int SERVO_SIDE1       = 0;      // Degrees — side 1 position
const int SERVO_SIDE2       = 180;    // Degrees — side 2 (after rotation)
const int SERVO_DELAY_MS    = 1000;   // Wait after servo move (ms)

// ─────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────
Servo hangerServo;

float currentX = 0.0;   // mm
float currentY = 0.0;   // mm
int   servoAngle = SERVO_SIDE1;

// ─────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────
void setup() {
    Serial.begin(115200);

    // Stepper pins
    pinMode(X_STEP_PIN, OUTPUT);
    pinMode(X_DIR_PIN,  OUTPUT);
    pinMode(Y_STEP_PIN, OUTPUT);
    pinMode(Y_DIR_PIN,  OUTPUT);
    pinMode(ENABLE_PIN, OUTPUT);
    digitalWrite(ENABLE_PIN, LOW);   // Enable motors

    // Relay pins — start OFF
    pinMode(STEAM_PIN,  OUTPUT);
    pinMode(VACUUM_PIN, OUTPUT);
    digitalWrite(STEAM_PIN,  LOW);
    digitalWrite(VACUUM_PIN, LOW);

    // Servo
    hangerServo.attach(SERVO_PIN);
    hangerServo.write(SERVO_SIDE1);

    Serial.println("OK");   // Signal ready to Python
}

// ─────────────────────────────────────────────
// STEPPER HELPERS
// ─────────────────────────────────────────────

/*
 * Move a single axis by a number of steps.
 * direction: HIGH = positive, LOW = negative
 * stepPin / dirPin: the axis pins
 * delayUs: microseconds between steps (controls speed)
 */
void moveAxis(int stepPin, int dirPin, long steps, bool direction, unsigned long delayUs) {
    digitalWrite(dirPin, direction ? HIGH : LOW);
    delayMicroseconds(1);   // Direction setup time
    for (long i = 0; i < steps; i++) {
        digitalWrite(stepPin, HIGH);
        delayMicroseconds(STEP_PULSE_US);
        digitalWrite(stepPin, LOW);
        delayMicroseconds(delayUs - STEP_PULSE_US);
    }
}

/*
 * Convert feed rate (mm/min) to step delay (µs/step).
 * feed_mm_min → mm/s → steps/s → µs/step
 */
unsigned long feedToDelayUs(float feed_mm_min) {
    if (feed_mm_min <= 0) feed_mm_min = DEFAULT_FEED;
    if (feed_mm_min > MAX_FEED) feed_mm_min = MAX_FEED;
    float steps_per_sec = (feed_mm_min / 60.0) * STEPS_PER_MM;
    return (unsigned long)(1000000.0 / steps_per_sec);
}

/*
 * Move end-effector to absolute position (X_mm, Y_mm).
 * Uses linear interpolation for simultaneous X+Y movement
 * (Bresenham-style step distribution for straight lines).
 */
void moveTo(float targetX, float targetY, float feed_mm_min) {
    float dx = targetX - currentX;
    float dy = targetY - currentY;

    long stepsX = (long)abs(round(dx * STEPS_PER_MM));
    long stepsY = (long)abs(round(dy * STEPS_PER_MM));
    bool dirX   = (dx >= 0);   // positive X = left direction
    bool dirY   = (dy <= 0);   // positive Y = up direction (image Y flipped)

    unsigned long delayUs = feedToDelayUs(feed_mm_min);

    // Set directions
    digitalWrite(X_DIR_PIN, dirX ? HIGH : LOW);
    digitalWrite(Y_DIR_PIN, dirY ? HIGH : LOW);
    delayMicroseconds(1);

    // Bresenham simultaneous stepping
    long stepsBig   = max(stepsX, stepsY);
    long stepsSmall = min(stepsX, stepsY);
    bool xIsBig     = (stepsX >= stepsY);

    long err = stepsBig / 2;

    for (long i = 0; i < stepsBig; i++) {
        // Step the primary axis
        int bigPin   = xIsBig ? X_STEP_PIN : Y_STEP_PIN;
        int smallPin = xIsBig ? Y_STEP_PIN : X_STEP_PIN;

        digitalWrite(bigPin, HIGH);

        // Step secondary axis when error accumulates
        err -= stepsSmall;
        if (err < 0) {
            digitalWrite(smallPin, HIGH);
            err += stepsBig;
        }

        delayMicroseconds(STEP_PULSE_US);
        digitalWrite(bigPin,   LOW);
        digitalWrite(smallPin, LOW);
        delayMicroseconds(delayUs - STEP_PULSE_US);
    }

    currentX = targetX;
    currentY = targetY;
}

// ─────────────────────────────────────────────
// COMMAND PARSER
// ─────────────────────────────────────────────

/*
 * Parse and execute one command string.
 * Returns true if command was recognised and executed.
 */
bool executeCommand(String cmd) {
    cmd.trim();

    // ── MOVE X<mm> Y<mm> F<feed> ──
    if (cmd.startsWith("MOVE")) {
        float x = currentX, y = currentY, f = DEFAULT_FEED;
        // Parse X
        int xi = cmd.indexOf('X');
        if (xi >= 0) x = cmd.substring(xi + 1).toFloat();
        // Parse Y
        int yi = cmd.indexOf('Y');
        if (yi >= 0) y = cmd.substring(yi + 1).toFloat();
        // Parse F
        int fi = cmd.indexOf('F');
        if (fi >= 0) f = cmd.substring(fi + 1).toFloat();
        moveTo(x, y, f);
        return true;
    }

    // ── STEAM_ON ──
    if (cmd == "STEAM_ON") {
        digitalWrite(STEAM_PIN, HIGH);
        return true;
    }

    // ── STEAM_OFF ──
    if (cmd == "STEAM_OFF") {
        digitalWrite(STEAM_PIN, LOW);
        return true;
    }

    // ── VACUUM_ON ──
    if (cmd == "VACUUM_ON") {
        digitalWrite(VACUUM_PIN, HIGH);
        return true;
    }

    // ── VACUUM_OFF ──
    if (cmd == "VACUUM_OFF") {
        digitalWrite(VACUUM_PIN, LOW);
        return true;
    }

    // ── SERVO_ROTATE — toggle between side 1 and side 2 ──
    if (cmd == "SERVO_ROTATE") {
        servoAngle = (servoAngle == SERVO_SIDE1) ? SERVO_SIDE2 : SERVO_SIDE1;
        hangerServo.write(servoAngle);
        delay(SERVO_DELAY_MS);
        return true;
    }

    // ── HOME — return to (0, 0) ──
    if (cmd == "HOME") {
        moveTo(0.0, 0.0, DEFAULT_FEED);
        return true;
    }

    return false;   // Unknown command
}

// ─────────────────────────────────────────────
// MAIN LOOP
// ─────────────────────────────────────────────
void loop() {
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();

        if (cmd.length() == 0) return;

        bool ok = executeCommand(cmd);

        if (ok) {
            Serial.println("OK");
        } else {
            Serial.println("ERR: Unknown command: " + cmd);
        }
    }
}