#define RESOLUTION_X 320
#define RESOLUTION_Y 240

#define CENTER_X 160
#define CENTER_Y 120

#define CHAR_COL 80
#define CHAR_ROW 60

#define CENTER_COL 40
#define CENTER_ROW 30

#define SDRAM_BASE            0xC0000000
#define FPGA_ONCHIP_BASE      0xC8000000
#define FPGA_CHAR_BASE        0xC9000000

#define LEDR_BASE             0xFF200000
#define HEX3_HEX0_BASE        0xFF200020
#define HEX5_HEX4_BASE        0xFF200030
#define SW_BASE               0xFF200040
#define KEY_BASE              0xFF200050
#define TIMER_BASE            0xFF202000
#define PIXEL_BUF_CTRL_BASE   0xFF203020
#define CHAR_BUF_CTRL_BASE    0xFF203030
#define PS2_BASE              0xFF200100

#define BOX_SIZE 5
#define SIZE 28

#define BORDER_LEFT 16
#define BORDER_RIGHT 302

#define BORDER_TOP 16
#define BORDER_BOTTOM 222

#include "updatedMain.h"

/************************************************************************************
*                                                                                   *
*   Graphic Arrays                                                                  *
*                                                                                   *
*************************************************************************************/

// Background and Title
#include "graphicHeaders/titleBackground.h"
#include "graphicHeaders/title.h"

// Button template
#include "graphicHeaders/button.h"
#include "graphicHeaders/buttonHover.h"

// Canvas Mode Icons
#include "graphicHeaders/eraser.h"
#include "graphicHeaders/pencil.h"

// Cursor Icon
#include "graphicHeaders/cursor.h"

/************************************************************************************
*                                                                                   *
*   Enums                                                                           *
*                                                                                   *
*************************************************************************************/

typedef enum {
    START = 0,
    LOAD = 1,
    MENU = 2,
    CANVAS = 3
} Page;

typedef enum {
    true = 1,
    false = 0
} bool;

typedef enum {
    NONE = 0,
    HIGHLIGHT_TRAIN = 1,
    HIGHLIGHT_DRAW = 2,
    DRAW = 3,
} Handle;

typedef enum {
    PENCIL,
    ERASE
} Mode;

typedef void (*page_draw_ptr)();
typedef void (*page_handle_ptr)();
typedef void (*handle_draw_ptr)();

typedef struct {
    int x;
    int y;
} Position;

typedef struct {
    int ySize;
    int xSize;
} Size;

/************************************************************************************
*                                                                                   *
*   Global Variables                                                                *
*                                                                                   *
*************************************************************************************/

int switchPageCount;
int handleNumber;
volatile int pixel_buffer_start;
Mode drawingMode;
int predictedNumber;
Page currentPage;

double drawArray[SIZE][SIZE];
Model model;

unsigned char seg[10] = {0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F};
unsigned char mousePackets[3] = {0, 0, 0}; // click = 0, x = 1, y = 2

int mouseX = 0, mouseY = 0;


/************************************************************************************
*                                                                                   *
*   Basic Drawing                                                                   *
*                                                                                   *
*************************************************************************************/

void write_char(int x, int y, char c) {
    // VGA character buffer
    volatile char * character_buffer = (char *) (FPGA_CHAR_BASE + (y<<7) + x);
    *character_buffer = c;
}

void plot_pixel(int x, int y, short int line_color) {
    *(short int *)(pixel_buffer_start + (y << 10) + (x << 1)) = line_color;
}

/************************************************************************************
*                                                                                   *
*   Render Functions                                                                *
*                                                                                   *
*************************************************************************************/

void drawComponent(Position pos, Size size, short int component[size.ySize][size.xSize]) {
    for(int y = 0; y < size.ySize; y++)
        for(int x = 0; x < size.xSize; x++)
            // Consider black pixels as transparent
            if(component[y][x] != 0x0)
                plot_pixel(pos.x + x, pos.y + y, component[y][x]);
}

void writeText(Position pos, char* text) {
    int x = pos.x;

    while(*text) {
        write_char(x, pos.y, *text);
        x++;
        text++;
    }
}

void drawCursor(int x, int y) {
    Size cursorSize = {sizeof(cursor) / sizeof(cursor[0]), sizeof(cursor[0]) / sizeof(cursor[0][0])};
    Position cursorPos = {x, y};
    drawComponent(cursorPos, cursorSize, cursor);
}

/************************************************************************************
*                                                                                   *
*   Page Rendering                                                                  *
*                                                                                   *
*************************************************************************************/

void startRender() {

    // Render Title
    Size titleSize = {sizeof(title) / sizeof(title[0]), sizeof(title[0]) / sizeof(title[0][0])};
    Position titlePos = {CENTER_X - titleSize.xSize / 2, BORDER_TOP + 40};
    drawComponent(titlePos, titleSize, title);
    
    // Render button
    Size buttonSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position buttonPos = {CENTER_X - buttonSize.xSize/2, RESOLUTION_Y * 2.0 / 3.0 - buttonSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(buttonPos, buttonSize, button);
    
    // Render train text
    char* text = "TRAIN MODEL";
    Position textPos = {CENTER_COL - 5, CHAR_ROW * 2.0 / 3.0};
    writeText(textPos, text);
    
}

void loadRender() {
    // Render title
    Size titleSize = {sizeof(title) / sizeof(title[0]), sizeof(title[0]) / sizeof(title[0][0])};
    Position titlePos = {CENTER_X - titleSize.xSize / 2, BORDER_TOP + 40};
    drawComponent(titlePos, titleSize, title);

    // Render loading text
    char* text = "LOADING...";
    Position textPos = {CHAR_COL / 2 - 5, CHAR_ROW * 2.0/3.0};
    writeText(textPos, text);
}

void menuRender() {
    // Render title
    Size titleSize = {sizeof(title) / sizeof(title[0]), sizeof(title[0]) / sizeof(title[0][0])};
    Position titlePos = {CENTER_X - titleSize.xSize / 2, BORDER_TOP + 40};
    drawComponent(titlePos, titleSize, title);
    
    // Render draw button
    Size drawSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position drawPos = {CENTER_X - drawSize.xSize/2, RESOLUTION_Y * 1.0 / 2.0 - drawSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(drawPos, drawSize, button);

    // Render exit button
    Size exitSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position exitPos = {CENTER_X - exitSize.xSize/2, RESOLUTION_Y * 2.0 / 3.0 - exitSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(exitPos, exitSize, button);

    // Render draw text
    char* drawText = "DRAW NOW";
    Position drawTextPos = {CHAR_COL / 2 - 4, CHAR_ROW * 1.0/2.0};
    writeText(drawTextPos, drawText);

    //Render exit text
    char* exitText = "EXIT";
    Position exitTextPos = {CHAR_COL / 2 - 2, CHAR_ROW * 2.0/3.0};
    writeText(exitTextPos, exitText);
}

void canvasRender() {

    // Draw the white background for the canvas
    Position canvasPos = {CENTER_X - SIZE * BOX_SIZE / 2, RESOLUTION_Y * 1.0 / 7.0};
    for(int y = 0; y < SIZE * BOX_SIZE; y++) 
        for(int x = 0; x < SIZE * BOX_SIZE; x++) 
            plot_pixel(canvasPos.x + x, canvasPos.y + y, 0xFFFF);    

    // Render back button
    Size backSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position backPos = {CENTER_X - backSize.xSize * 3.0 / 2.0 + 2, RESOLUTION_Y * 9.0 / 11.0 - backSize.ySize / 2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(backPos, backSize, button);
	
	// Render predict button
    Size predictSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position predictPos = {CENTER_X + predictSize.xSize * 1.0 / 2.0 - 2, RESOLUTION_Y * 9.0 / 11.0 - predictSize.ySize / 2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(predictPos, predictSize, button);

	// Render back text
    char* backText = "BACK TO MENU";
    Position backTextPos = {CHAR_COL / 2 - 6 - backSize.xSize * CHAR_COL / RESOLUTION_X, CHAR_ROW * 9.0/11.0};
    writeText(backTextPos, backText);
	
	// Render predict text
    char* predictText = "RECOGNIZE IT";
    Position predictTextPos = {CHAR_COL / 2 - 6 + predictSize.xSize * CHAR_COL / RESOLUTION_X, CHAR_ROW * 9.0/11.0};
    writeText(predictTextPos, predictText);


    // Render icon of current drawing mode
    Size modeSize; 
    
    if(drawingMode == PENCIL) {
        modeSize.ySize = sizeof(pencil) / sizeof(pencil[0]);
        modeSize.xSize = sizeof(pencil[0])/sizeof(pencil[0][0]);
    } else if(drawingMode == ERASE) {
        modeSize.ySize = sizeof(eraser) / sizeof(eraser[0]);
        modeSize.xSize = sizeof(eraser[0]) / sizeof(eraser[0][0]);
    }
    
    Position modePos = {CENTER_X - modeSize.xSize / 2, RESOLUTION_Y * 9.0 / 11.0 - modeSize.ySize / 2 + RESOLUTION_Y / CHAR_ROW / 2};

    if(drawingMode == PENCIL)
        drawComponent(modePos, modeSize, pencil);
    else if(drawingMode == ERASE) 
        drawComponent(modePos, modeSize, eraser);
}

page_draw_ptr drawPage[] = {startRender, loadRender, menuRender, canvasRender};


/************************************************************************************
*                                                                                   *
*   Page Handlers                                                                   *
*                                                                                   *
*************************************************************************************/

void startHandle() {
    /*
        No Handle = 0
        Train Button Hover = 1
        Train Button No Hover = 2
        Train Button Click = 3
    */
    handleNumber = 0;

    if()

}

void loadHandle() {
    /*
        No Handle = 0
        Load model = 4
    */
   handleNumber = 4;
}

void menuHandle() {
    /*
        No Handle = 0
        Draw Button Hover = 5
        Draw Button No Hover = 6
        Exit Button Hover = 7
        Exit Button No Hover = 8
        Draw Button Click = 9
        Exit Button Click = 10
    */
    handleNumber = 0;

}

void canvasHandle() {
    /*
        No Handle = 0
        Draw Cursor = 11
        Back Button Hover = 12
        Back Button No Hover = 13
        Recognize Button Hover = 14
        Recognize Button No Hover = 15
        Back Button Click = 16
        Recognize Button Click = 17
        Mode Button Click = 18
    */
    handleNumber = 0;

}

page_handle_ptr handlePage[] = {startHandle, loadHandle, menuHandle, canvasHandle};

/************************************************************************************
*                                                                                   *
*   Handle Event Rendering                                                          *
*                                                                                   *
*************************************************************************************/


/* Global Handle */
void noHandle() {
    return;
}

/***************************************************
*   Start Handles                                  *
****************************************************/

void trainButtonHover() {
    // Render highlighted button
    Size buttonSize = {sizeof(buttonHover) / sizeof(buttonHover[0]), sizeof(buttonHover[0]) / sizeof(buttonHover[0][0])};
    Position buttonPos = {CENTER_X - buttonSize.xSize/2, RESOLUTION_Y * 2.0 / 3.0 - buttonSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(buttonPos, buttonSize, buttonHover);
}

void trainButtonNoHover() {
    // Render normal button
    Size buttonSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position buttonPos = {CENTER_X - buttonSize.xSize/2, RESOLUTION_Y * 2.0 / 3.0 - buttonSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(buttonPos, buttonSize, button);
}

void trainButtonClick() {
    currentPage = LOAD;
}

/***************************************************
*   Load Handles                                   *
****************************************************/

void loadModel() {
    
    srand(time(0));

    Model model;

    initializeModel(&model, SIZE);

    addLayer(&model, LINEAR, 50, leakReLuVector, leakReLuGradient);

    addLayer(&model, LINEAR, 10, softMaxVector, softMaxGradient);

    setupModel(&model, RandomInitialization, crossEntropyGradientWithSoftmax);

    int batchSize = 10;
    int epochs = 2;
    double learningRate = 0.1;

    trainModel(&model,
                NUM_TRAIN, SIZE, train_image, train_label, 
                batchSize, epochs, learningRate,
                NUM_TEST, test_image, test_label);
    
    currentPage = MENU;
}


/***************************************************
*   Menu Handles                                   *
****************************************************/

void drawButtonHover() {
    // Render draw button
    Size drawSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position drawPos = {CENTER_X - drawSize.xSize/2, RESOLUTION_Y * 1.0 / 2.0 - drawSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(drawPos, drawSize, button);
}

void drawButtonNoHover() {
    // Render draw hover button
    Size drawSize = {sizeof(buttonHover) / sizeof(buttonHover[0]), sizeof(buttonHover[0]) / sizeof(buttonHover[0][0])};
    Position drawPos = {CENTER_X - drawSize.xSize/2, RESOLUTION_Y * 1.0 / 2.0 - drawSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(drawPos, drawSize, buttonHover);
}

void exitButtonHover() {
    // Render exit button
    Size exitSize = {sizeof(buttonHover) / sizeof(buttonHover[0]), sizeof(buttonHover[0]) / sizeof(buttonHover[0][0])};
    Position exitPos = {CENTER_X - exitSize.xSize/2, RESOLUTION_Y * 2.0 / 3.0 - exitSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(exitPos, exitSize, buttonHover);
}

void exitButtonNoHover() {
    // Render exit button
    Size exitSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position exitPos = {CENTER_X - exitSize.xSize/2, RESOLUTION_Y * 2.0 / 3.0 - exitSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(exitPos, exitSize, button);
}

void drawButtonClick() {
    currentPage = CANVAS;
}

void exitButtonClick() {
    exit(1);
} 

/***************************************************
*   Canvas Handles                                 *
****************************************************/

void drawCanvas() {

} 

void backButtonHover() {
    // Render back button
    Size backSize = {sizeof(buttonHover) / sizeof(buttonHover[0]), sizeof(buttonHover[0]) / sizeof(buttonHover[0][0])};
    Position backPos = {CENTER_X - backSize.xSize * 3.0 / 2.0 + 2, RESOLUTION_Y * 9.0 / 11.0 - backSize.ySize / 2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(backPos, backSize, buttonHover);
}


void backButtonNoHover() {
    // Render back button
    Size backSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position backPos = {CENTER_X - backSize.xSize * 3.0 / 2.0 + 2, RESOLUTION_Y * 9.0 / 11.0 - backSize.ySize / 2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(backPos, backSize, button);
}

void recognizeButtonHover() {
    // Render predict button
    Size predictSize = {sizeof(buttonHover) / sizeof(buttonHover[0]), sizeof(buttonHover[0]) / sizeof(buttonHover[0][0])};
    Position predictPos = {CENTER_X + predictSize.xSize * 1.0 / 2.0 - 2, RESOLUTION_Y * 9.0 / 11.0 - predictSize.ySize / 2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(predictPos, predictSize, buttonHover);
}

void recognizeButtonNoHover() {
    // Render predict button
    Size predictSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position predictPos = {CENTER_X + predictSize.xSize * 1.0 / 2.0 - 2, RESOLUTION_Y * 9.0 / 11.0 - predictSize.ySize / 2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(predictPos, predictSize, button);
}

void backButtonClick() {

}

void recognizeButtonClick() {
    /*
        predictedNumber = predictModel(&model, drawArray);
    */
}

void modeButtonClick() {

}

/***************************************************
*   Handle Function Array                          *
****************************************************/

handle_draw_ptr handleRender[] = {noHandle, 
                                  trainButtonHover, trainButtonNoHover, trainButtonClick, 
                                  loadModel,
                                  drawButtonHover, drawButtonNoHover, exitButtonHover, exitButtonNoHover, drawButtonClick, exitButtonClick, 
                                  drawCanvas, backButtonHover, backButtonNoHover, recognizeButtonHover, recognizeButtonNoHover,
                                  backButtonClick, recognizeButtonClick, modeButtonClick};

/************************************************************************************
*                                                                                   *
*   Buffering                                                                       *
*                                                                                   *
*************************************************************************************/


void wait_for_vsync() {
	volatile int* buffer = PIXEL_BUF_CTRL_BASE;
    
	*buffer = 1;
	register int status = *(buffer+3);
	
	while((status & 0x01) != 0)
		status = *(buffer+3);
}

void clear_screen() {
	for(int x = BORDER_LEFT; x < BORDER_RIGHT; x++) 
		for(int y = BORDER_TOP; y < BORDER_BOTTOM; y++)
			plot_pixel(x, y, 0x0);	
}

void clear_character() {
    for(int x = 0; x < CHAR_COL; x++)
        for(int y = 0; y < CHAR_ROW; y++)
            write_char(x, y, ' ');
}

void drawBackground() {
    for(int i = 0; i < 240; i++)
        for(int j = 0; j < 320; j++)
            plot_pixel(j, i, titleBackground[i][j]);
}

/************************************************************************************
*                                                                                   *
*   HEX Segment Functions                                                           *
*                                                                                   *
*************************************************************************************/

void displayResult(int num) {
	*HEX3_HEX0_BASE = seg[num];
}

/************************************************************************************
*                                                                                   *
*   Mouse Input Functions                                                           *
*                                                                                   *
*************************************************************************************/

void mouseInput() {
	
  	volatile int * PS2_ptr = (int *) PS2_BASE;  // PS/2 port address
    unsigned char packet_complete = 0;

    static enum mouse_status status = NOT_REPORTING;
    
	int PS2_data, RVALID;
    
    PS2_data = *(PS2_ptr);	// read the Data register in the PS/2 port

    while (1) {
        RVALID = (PS2_data & 0x8000);	// extract the RVALID field
        if (!RVALID) break;

        mousePackets[0] = mousePackets[1];
        mousePackets[1] = mousePackets[2];
        mousePackets[2] = PS2_data & 0xFF;

        if (mousePackets[1] == 0xAA && mousePackets[2] == 0x00) {
            *(PS2_ptr) = 0xF4;
            status = AWAITING_ACK;
        }
        else if (status == AWAITING_ACK && mousePackets[2] == 0xFA) {
            status = REPORTING;
            struct event_t event = {E_MOUSE_ENABLED, {.mouse_enabled = {}}};
            event_queue_push(event, "mouse enabled");
        }
        continue;
        packet_complete = (packet_complete + 1) % 3;

        if (packet_complete != 0) continue;

        struct mouse_state_t prev_state = mouse_state;
        update_mouse_state(mousePackets);
        post_mouse_events(prev_state);
    }

    drawCursor(x, y);
}

void update_mouse_state(char mousePackets[3]) {
    mouse_state.left_clicked = mousePackets[0] & 0b001;
    mouse_state.right_clicked = (mousePackets[0] & 0b010) >> 1;

    struct {
        signed short x : 9;
        signed short y : 9;
    } pos;
    pos.x = (((signed short) mousePackets[0] & 0x10) << 4) | (signed short) mousePackets[1];
    pos.y = (((signed short) mousePackets[0] & 0x20) << 3) | (signed short) mousePackets[2];

    signed short dx = pos.x;
    signed short dy = pos.y;
    
    mouse_state.x += dx * MOUSE_SENSITIVITY;
    mouse_state.y += dy * MOUSE_SENSITIVITY;

    if(mouse_state.x < 0) mouse_state.x = 0;
    else if(mouse_state.x >= RESOLUTION_X) mouse_state.x = RESOLUTION_X - 1;
    if(mouse_state.y < 0) mouse_state.y = 0;
    else if(mouse_state.y >= RESOLUTION_Y) mouse_state.y = RESOLUTION_Y - 1;
}

void post_mouse_events(struct mouse_state_t prev_state) {
    struct event_t event;

    struct e_mouse_move movement;
    int moved = prev_state.x != mouse_state.x || prev_state.y != mouse_state.y;
    movement.x = mouse_state.x * moved;
    movement.y = mouse_state.y * moved;

    if(movement.x != 0 || movement.y != 0) {
        event.type = E_MOUSE_MOVE;
        event.data.mouse_move = movement;
        event_queue_push(event, "mouse move");
    }

    struct e_mouse_button button_down = {0,0,0}, button_up = {0,0,0};

    (mouse_state.left_clicked ? &button_down : &button_up) -> left = prev_state.left_clicked != mouse_state.left_clicked;
    (mouse_state.right_clicked ? &button_down : &button_up) -> right = prev_state.right_clicked != mouse_state.right_clicked;

    if (button_up.left || button_up.right) {
        event.type = E_MOUSE_BUTTON_UP;
        event.data.mouse_button_up = button_up;
        event_queue_push(event, "mouse button up");
    }

    if (button_down.left || button_down.right) {
        event.type = E_MOUSE_BUTTON_DOWN;
        event.data.mouse_button_down = button_down;
        event_queue_push(event, "mouse button down");
    }
}


/************************************************************************************
*                                                                                   *
*   Main Function                                                                   *
*                                                                                   *
*************************************************************************************/


int main() {

    /* Set up page */
    currentPage = LOAD;
    switchPageCount = 0;
    handleNumber = 0;
    drawingMode = DRAW;

    /* Set up buffers */
    volatile int * pixel_ctrl_ptr = (int*)PIXEL_BUF_CTRL_BASE;
    
    clear_character();

    *(pixel_ctrl_ptr + 1) = FPGA_ONCHIP_BASE; 
    wait_for_vsync();
    
    pixel_buffer_start = *pixel_ctrl_ptr;
    drawBackground();
    drawPage[currentPage]();

    *(pixel_ctrl_ptr + 1) = SDRAM_BASE;
    
    pixel_buffer_start = *(pixel_ctrl_ptr + 1);
    drawBackground(); 
    drawPage[currentPage]();

    
	
    while (1)
    {
        clear_screen();
        clear_character();
        // Handle user input via polling depending on page and changes handleNumber if event handle occured 
        handlePage[currentPage]();

        // Draw the next page on back buffer. Draws again when swapped buffers.
        if(switchPageCount < 2) {
            clear_screen();
            drawPage[currentPage]();
            switchPageCount++;
        }

        // Render any event handles that occured from handlePage
        handleRender[handleNumber]();
        
        handlePage[currentPage]();
        
        wait_for_vsync(); // swap front and back buffers on VGA vertical sync
        pixel_buffer_start = *(pixel_ctrl_ptr + 1); // new back buffer
    }
}

void displayResult(int num) {
	HEX3_HEX0_BASE = seg[num];
}
