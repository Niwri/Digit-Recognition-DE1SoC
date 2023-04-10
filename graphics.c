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
#define HEX3_HEX0_BASE        (volatile long*) 0xFF200020
#define HEX5_HEX4_BASE        0xFF200030
#define SW_BASE               0xFF200040
#define KEY_BASE              0xFF200050
#define TIMER_BASE            0xFF202000
#define PIXEL_BUF_CTRL_BASE   0xFF203020
#define CHAR_BUF_CTRL_BASE    0xFF203030
#define PS2_BASE              0xFF200100

#define PS2_IRQ               79

#define MPCORE_GIC_CPUIF       
#define ICCEOIR                0xFFFEC110
#define ICCIAR                 0xFFFEC10C

#define BOX_SIZE 5
#define CANVAS_SIZE 28

#define BORDER_LEFT 16
#define BORDER_RIGHT 302

#define BORDER_TOP 16
#define BORDER_BOTTOM 222

#define SENSITIVITY 0.20

#include "updatedModel.h"
#include "graphics.h"

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
*   Global Variables                                                                *
*                                                                                   *
*************************************************************************************/

int switchPageCount;
volatile int pixel_buffer_start;
Mode drawingMode;
int predictedNumber;
Page currentPage;

double drawArray[CANVAS_SIZE][CANVAS_SIZE];
Model model;

unsigned char seg[10] = {0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F};

// To keep track of mouse movements and events
unsigned char mousePackets[3] = {0, 0, 0}; // click = 0, x = 1, y = 2
Position mouse = { BORDER_LEFT, BORDER_TOP+9};
bool leftClick = false;

// To store every handle event to process 
int handleNum[3] = {-1, -1, -1}; 
int numOfHandles = 0;

bool loadTrain = false;

Status currentStatus = DEFAULT;

int switchPageCount = 0;
bool switchPage = false;

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
*   Helper Functions                                                                *
*                                                                                   *
*************************************************************************************/

bool mouseIsInside(int minX, int maxX, int minY, int maxY) {
    if(minX <= mouse.x && mouse.x <= maxX && minY <= mouse.y && mouse.y <= maxY) 
        return true;
	
    return false;
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

void drawCursor(int mx, int my) {
    Size cursorSize = {sizeof(cursor) / sizeof(cursor[0]), sizeof(cursor[0]) / sizeof(cursor[0][0])};
    for(int y = 0; y < cursorSize.ySize; y++)
        for(int x = 0; x < cursorSize.xSize; x++)
            if(cursor[y][x] != 0x0)
                plot_pixel(mx + x, my - cursorSize.ySize + y, cursor[y][x]);
}

void removeCursor(Position mousePosition) {
    Size cursorSize = {sizeof(cursor) / sizeof(cursor[0]), sizeof(cursor[0]) / sizeof(cursor[0][0])};
    for(int y = 0; y < cursorSize.ySize; y++)
        for(int x = 0; x < cursorSize.xSize; x++)
            if(cursor[y][x] != 0x0)
                plot_pixel(mousePosition.x + x, mousePosition.y - cursorSize.ySize + y, 0x0);
}

void clearCanvas() {
    for(int i = 0; i < CANVAS_SIZE; i++)
        for(int j = 0; j < CANVAS_SIZE; j++)
            drawArray[i][j] = 0;
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
    Position canvasPos = {CENTER_X - CANVAS_SIZE * BOX_SIZE / 2, RESOLUTION_Y * 1.0 / 7.0};
	
	for(int my = 0; my < CANVAS_SIZE; my++)
        for(int mx = 0; mx < CANVAS_SIZE; mx++)
            for(int y = 0; y < BOX_SIZE; y++) 
                for(int x = 0; x < BOX_SIZE; x++) {
                    short int RBcharacter = 31 - (unsigned short)(drawArray[my][mx] * 31);
                    short int Gcharacter = 63 - (unsigned short)(drawArray[my][mx] * 63);
					short int color = RBcharacter << 11 | (Gcharacter << 5) | RBcharacter;
                    plot_pixel(canvasPos.x + mx * BOX_SIZE + x, canvasPos.y + my * BOX_SIZE + y, color);  
                }  

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

    // Check if mouse is inside train button
    Size buttonSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position buttonPos = {CENTER_X - buttonSize.xSize/2, RESOLUTION_Y * 2.0 / 3.0 - buttonSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};

    if(mouseIsInside(buttonPos.x, buttonPos.x + buttonSize.xSize, buttonPos.y, buttonPos.y + buttonSize.ySize) == true) {
        
        handleNum[0] = 1;
        // Check if mouse clicked on train button
        if(leftClick) {
            handleNum[0] = 3;
        }
    } else {
        handleNum[0] = 2;
    }

    numOfHandles = 1;
}

void loadHandle() {
    /*
        No Handle = 0
        Load model = 4
    */
   // Train Model
   handleNum[0] = 4;
   numOfHandles = 1;
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

    // Check if user is hovering over draw button
    Size drawSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position drawPos = {CENTER_X - drawSize.xSize/2, RESOLUTION_Y * 1.0 / 2.0 - drawSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};

    if(mouseIsInside(drawPos.x, drawPos.x + drawSize.xSize, drawPos.y, drawPos.y + drawSize.ySize) == true) {
        handleNum[0] = 5;

        // Check if user clicked on draw button
        if(leftClick) {
            handleNum[0] = 9;
            
        }
    } else {
        handleNum[0] = 6;
    }

    // Check if user is hovering over exit button
    Size exitSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position exitPos = {CENTER_X - exitSize.xSize/2, RESOLUTION_Y * 2.0 / 3.0 - exitSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};

    if(mouseIsInside(exitPos.x, exitPos.x + exitSize.xSize, exitPos.y, exitPos.y + exitSize.ySize) == true) {
        handleNum[1] = 7;

        // Check if user clicked on exit button
        if(leftClick) {
            handleNum[1] = 10;
        }
    } else {
        handleNum[1] = 8;
    }

    numOfHandles = 2;
}

void canvasHandle() {
    /*
        No Handle = 0
        Draw Canvas = 11
        Back Button Hover = 12
        Back Button No Hover = 13
        Recognize Button Hover = 14
        Recognize Button No Hover = 15
        Back Button Click = 16
        Recognize Button Click = 17
        Mode Button Click = 18
    */
    
    // Check if mouse left-clicked in canvas for drawing
    Position canvasPos = {CENTER_X - CANVAS_SIZE * BOX_SIZE / 2, RESOLUTION_Y * 1.0 / 7.0};

    if(mouseIsInside(canvasPos.x, canvasPos.x + CANVAS_SIZE * BOX_SIZE, canvasPos.y, canvasPos.y + CANVAS_SIZE * BOX_SIZE) == true) {
        if(leftClick) {
            handleNum[0] = 11;
            numOfHandles++;
        }
    }

    // Check if user is hovering over back button
    Size backSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position backPos = {CENTER_X - backSize.xSize * 3.0 / 2.0 + 2, RESOLUTION_Y * 9.0 / 11.0 - backSize.ySize / 2 + RESOLUTION_Y / CHAR_ROW / 2};
    if(mouseIsInside(backPos.x, backPos.x + backSize.xSize, backPos.y, backPos.y + backSize.ySize) == true) {
        handleNum[numOfHandles] = 12;

        // Check if user clicked on back button
        if(leftClick) {
            handleNum[numOfHandles] = 16;
        }
    } else {
        handleNum[numOfHandles] = 13;
    }
    numOfHandles++;
	
	// Check if user is hovering over predict button
    Size predictSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position predictPos = {CENTER_X + predictSize.xSize * 1.0 / 2.0 - 2, RESOLUTION_Y * 9.0 / 11.0 - predictSize.ySize / 2 + RESOLUTION_Y / CHAR_ROW / 2};
    if(mouseIsInside(predictPos.x, predictPos.x + predictSize.xSize, predictPos.y, predictPos.y + predictSize.ySize) == true) {
        handleNum[numOfHandles] = 14;

        // Check if user clicked on predict button
        if(leftClick) {
            handleNum[numOfHandles] = 17;
        }
    } else {
        handleNum[numOfHandles] = 15;
    }
    numOfHandles++;
    
    // Check if user clicked on mode button to switch drawing mode
    Size modeSize; 
    
    if(drawingMode == PENCIL) {
        modeSize.ySize = sizeof(pencil) / sizeof(pencil[0]);
        modeSize.xSize = sizeof(pencil[0])/sizeof(pencil[0][0]);
    } else if(drawingMode == ERASE) {
        modeSize.ySize = sizeof(eraser) / sizeof(eraser[0]);
        modeSize.xSize = sizeof(eraser[0]) / sizeof(eraser[0][0]);
    }
    
    Position modePos = {CENTER_X - modeSize.xSize / 2, RESOLUTION_Y * 9.0 / 11.0 - modeSize.ySize / 2 + RESOLUTION_Y / CHAR_ROW / 2};

    if(mouseIsInside(modePos.x, modePos.x + modeSize.xSize, modePos.y, modePos.y + modeSize.ySize) == true) {
        if(leftClick) {
            handleNum[numOfHandles] = 18;
            numOfHandles++;
        }
    }
    
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
    switchPage = true;
}

/***************************************************
*   Load Handles                                   *
****************************************************/

void loadModel() {
    
    if(loadTrain == false) {
        loadTrain = true;
        return;
    }
    srand(time(0));

    Model model;

    initializeModel(&model, SIZE);

    addLayer(&model, LINEAR, 50, leakReLuVector, leakReLuGradient);

    addLayer(&model, LINEAR, 10, softMaxVector, softMaxGradient);

    setupModel(&model, RandomInitialization, crossEntropyGradientWithSoftmax);

    int batchSize = 100;
    int epochs = 13;
    double learningRate = 0.1;

    trainModel(&model,
                NUM_TRAIN, SIZE, train_image, train_label, 
                batchSize, epochs, learningRate,
                NUM_TEST, test_image, test_label);
    
    currentPage = MENU;
    switchPage = true;
}


/***************************************************
*   Menu Handles                                   *
****************************************************/

void drawButtonHover() {
    // Render draw hover button
    Size drawSize = {sizeof(buttonHover) / sizeof(buttonHover[0]), sizeof(buttonHover[0]) / sizeof(buttonHover[0][0])};
    Position drawPos = {CENTER_X - drawSize.xSize/2, RESOLUTION_Y * 1.0 / 2.0 - drawSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(drawPos, drawSize, buttonHover);

}

void drawButtonNoHover() {    // Render draw button
    Size drawSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position drawPos = {CENTER_X - drawSize.xSize/2, RESOLUTION_Y * 1.0 / 2.0 - drawSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(drawPos, drawSize, button);
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
    switchPage = true;
}

void exitButtonClick() {
    exit(1);
} 

/***************************************************
*   Canvas Handles                                 *
****************************************************/

void drawCanvasArray() {
    // Draw the white background for the canvas
    Position canvasPos = {CENTER_X - CANVAS_SIZE * BOX_SIZE / 2, RESOLUTION_Y * 1.0 / 7.0};

    int xCoord = (mouse.x - canvasPos.x) / BOX_SIZE;
    int yCoord = (mouse.y - canvasPos.y) / BOX_SIZE;

    if (xCoord < 0 || yCoord < 0) return;

    if (drawingMode == PENCIL) {
        drawArray[yCoord][xCoord] = 1.0;
        if (xCoord < 27) {
            drawArray[yCoord][xCoord + 1] += 0.5;
            if (drawArray[yCoord][xCoord + 1] > 1) drawArray[yCoord][xCoord + 1] = 1;
        }

        if (xCoord > 0) {
            drawArray[yCoord][xCoord - 1] += 0.5;
            if (drawArray[yCoord][xCoord - 1] > 1) drawArray[yCoord][xCoord - 1] = 1;
        }

        if (yCoord < 27) {
            drawArray[yCoord + 1][xCoord] += 0.5;
            if (drawArray[yCoord + 1][xCoord] > 1) drawArray[yCoord + 1][xCoord] = 1;
        }

        if (yCoord > 0) {
            drawArray[yCoord - 1][xCoord] += 0.5;
            if (drawArray[yCoord - 1][xCoord] > 1) drawArray[yCoord - 1][xCoord] = 1;
        }
    
    }
    else {
        drawArray[yCoord][xCoord] = 0.0;
        if (xCoord < 27) {
            drawArray[yCoord][xCoord + 1] -= 0.5;
            if (drawArray[yCoord][xCoord + 1] < 0) drawArray[yCoord][xCoord + 1] = 0;
        }

        if (xCoord > 0) {
            drawArray[yCoord][xCoord - 1] -= 0.5;
            if (drawArray[yCoord][xCoord - 1] < 0) drawArray[yCoord][xCoord - 1] = 0;
        }

        if (yCoord < 27) {
            drawArray[yCoord + 1][xCoord] -= 0.5;
            if (drawArray[yCoord + 1][xCoord] < 0) drawArray[yCoord + 1][xCoord] = 0;
        }

        if (yCoord > 0) {
            drawArray[yCoord - 1][xCoord] -= 0.5;
            if (drawArray[yCoord - 1][xCoord] < 0) drawArray[yCoord - 1][xCoord] = 0;
        }
    }
        
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
    clearCanvas();
    currentPage = MENU;
    switchPage = true;
}

void recognizeButtonClick() {
    double featureArray[SIZE] = {0};
    for(int y = 0; y < CANVAS_SIZE; y++) 
        for(int x = 0; x < CANVAS_SIZE; x++)
            featureArray[y * CANVAS_SIZE + x] = drawArray[y][x];
    printf("Recognizing...\n");

    printf("Processed image:\n");
    for (int i=0; i<784; i++) {
        printf("%1.1f ", test_image[testNum][i]);
        if ((i+1) % 28 == 0) putchar('\n');
    }

    displayResult(predictModel(&model, SIZE, featureArray));
}

void modeButtonClick() {
    drawingMode = drawingMode == PENCIL ? ERASE : PENCIL;
}

/***************************************************
*   Handle Function Array                          *
****************************************************/

handle_draw_ptr handleRender[] = {noHandle, 
                                  trainButtonHover, trainButtonNoHover, trainButtonClick, 
                                  loadModel,
                                  drawButtonHover, drawButtonNoHover, exitButtonHover, exitButtonNoHover, drawButtonClick, exitButtonClick, 
                                  drawCanvasArray, backButtonHover, backButtonNoHover, recognizeButtonHover, recognizeButtonNoHover,
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
    printf("Received Result:  %d\n", num);
	*HEX3_HEX0_BASE = seg[num];
}

/************************************************************************************
*                                                                                   *
*   Mouse Input Functions                                                           *
*                                                                                   *
*************************************************************************************/

void mouseInput() {
    volatile int * PS2_ptr = (int *) PS2_BASE;
    unsigned char packet_complete = 0;
    int PS2_data, RVALID;

    int numOfBytes = 0;

    while (numOfBytes < 3) {
        PS2_data = *(PS2_ptr);
        RVALID = (PS2_data & 0x8000);

        if (RVALID) {

            mousePackets[0] = mousePackets[1];
            mousePackets[1] = mousePackets[2];
            mousePackets[2] = PS2_data & 0xFF;

            if(currentStatus == REPORTING)
                numOfBytes++;


            if(currentStatus != REPORTING && mousePackets[1] == (unsigned char)0xAA && mousePackets[2] == (unsigned char)0x00) {
                currentStatus = WAIT_ACKNOWLEDGE;
                *(PS2_ptr) = 0xF4;
            } 

            if(currentStatus == WAIT_ACKNOWLEDGE && mousePackets[2] == 0xFA) {
                currentStatus = REPORTING;
                continue;
            }
        }

    }

    struct {
        signed int x : 9;
        signed int y : 9;
    } signedPos;

    signedPos.x = ((mousePackets[0] & 0b10000) << 4) | (mousePackets[1]);
    signedPos.y = ((mousePackets[0] & 0b100000) << 3) | (mousePackets[2]);


    mouse.x += signedPos.x * SENSITIVITY;
    mouse.y -= signedPos.y * SENSITIVITY;
    
    if(mouse.x > BORDER_RIGHT - 9)
        mouse.x = BORDER_RIGHT - 9;
    if(mouse.y > BORDER_BOTTOM)
        mouse.y = BORDER_BOTTOM;

    if(mouse.x < BORDER_LEFT)
        mouse.x = BORDER_LEFT;
    if(mouse.y < BORDER_TOP + 9)
        mouse.y = BORDER_TOP + 9;

    leftClick = mousePackets[0] & 0b1;
}


/************************************************************************************
*                                                                                   *
*   Interrupt Configurations                                                        *
*                                                                                   *
*************************************************************************************/

void config_GIC(void) {
    config_interrupt (79, 1); // configure the FPGA KEYs interrupt (73)
    // Set Interrupt Priority Mask Register (ICCPMR). Enable interrupts of all
    // priorities
    *((int *) 0xFFFEC104) = 0xFFFF;
    // Set CPU Interface Control Register (ICCICR). Enable signaling of
    // interrupts
    *((int *) 0xFFFEC100) = 1;
    // Configure the Distributor Control Register (ICDDCR) to send pending
    // interrupts to CPUs
    *((int *) 0xFFFED000) = 1;
}

void config_PS2(void) {
    volatile int* ptr = 0xFF200100;
    *(ptr + 0x1) = 0x1;
}

void set_A9_IRQ_stack(void) {
    int stack, mode;
    stack = 0xFFFFFFFF - 7; // top of A9 onchip memory, aligned to 8 bytes
    /* change processor to IRQ mode with interrupts disabled */
    mode = 0b11010010;
    asm("msr cpsr, %[ps]" : : [ps] "r"(mode));
    /* set banked stack pointer */
    asm("mov sp, %[ps]" : : [ps] "r"(stack));
    /* go back to SVC mode before executing subroutine return! */
    mode =  0b11010011;

    asm("msr cpsr, %[ps]" : : [ps] "r"(mode));
}

void enable_A9_interrupts(void) {
    int status =  0b01010011;
    asm("msr cpsr, %[ps]" : : [ps] "r"(status));
}

void disable_A9_interrupts(void) {
    int status = 0b11010011;
    asm("msr cpsr, %[ps]" : : [ps] "r"(status));
}

void config_interrupt(int N, int CPU_target) {
    int reg_offset, index, value, address;
    /* Configure the Interrupt Set-Enable Registers (ICDISERn).
    * reg_offset = (integer_div(N / 32) * 4
    * value = 1 << (N mod 32) */
    reg_offset = (N >> 3) & 0xFFFFFFFC;
    index = N & 0x1F;
    value = 0x1 << index;
    address = 0xFFFED100 + reg_offset;
    /* Now that we know the register address and value, set the appropriate bit */
    *(int *)address |= value;
    /* Configure the Interrupt Processor Targets Register (ICDIPTRn)
    * reg_offset = integer_div(N / 4) * 4
    * index = N mod 4 */
    reg_offset = (N & 0xFFFFFFFC);
    index = N & 0x3;
    address = 0xFFFED800 + reg_offset + index;
    /* Now that we know the register address and value, write to (only) the
    * appropriate byte */
    *(char *)address = (char)CPU_target;
}


/************************************************************************************
*                                                                                   *
*   Interrupt Handler FUnctions                                                     *
*                                                                                   *
*************************************************************************************/

// Define the IRQ exception handler
void __attribute__((interrupt)) __cs3_isr_irq(void)
{
    // Read the ICCIAR from the processor interface
    int address = ICCIAR;
    int int_ID = *((int *)address);

    if (int_ID == PS2_IRQ) // check if interrupt is from the HPS timer
        mouseInput();
    else
        while (1); // if unexpected, then stay here

    // Write to the End of Interrupt Register (ICCEOIR)
    address = ICCEOIR;
    *((int *)address) = int_ID;
    return;
}

// Define the remaining exception handlers
void __attribute__((interrupt)) __cs3_reset(void)
{
while (1);
}
void __attribute__((interrupt)) __cs3_isr_undef(void)
{
while (1);
}
void __attribute__((interrupt)) __cs3_isr_swi(void)
{
while (1);
}
void __attribute__((interrupt)) __cs3_isr_pabort(void)
{
while (1);
}
void __attribute__((interrupt)) __cs3_isr_dabort(void)
{
while (1);
}
void __attribute__((interrupt)) __cs3_isr_fiq(void)
{
while (1);
}


/************************************************************************************
*                                                                                   *
*   Main Function                                                                   *
*                                                                                   *
*************************************************************************************/


int main() {
    disable_A9_interrupts();
    set_A9_IRQ_stack();
    config_GIC();
    config_PS2();
    enable_A9_interrupts();

    /* Set up page */
    currentPage = START;
    switchPageCount = 0;
    numOfHandles = 0;
    drawingMode = PENCIL;

    /* Set up buffers */
    volatile int * pixel_ctrl_ptr = (int*) PIXEL_BUF_CTRL_BASE;
    
    clear_character();

    *(pixel_ctrl_ptr + 1) = FPGA_ONCHIP_BASE; 
    wait_for_vsync();
    
    pixel_buffer_start = *pixel_ctrl_ptr;
    drawBackground();
    clear_screen();
    drawPage[currentPage]();

    *(pixel_ctrl_ptr + 1) = SDRAM_BASE;
    
    pixel_buffer_start = *(pixel_ctrl_ptr + 1);
    drawBackground(); 
    clear_screen();
    drawPage[currentPage]();

    // Previous mouse position of first buffer (FPGA_ONCHIP_BASE)
    Position mousePrevOne = {mouse.x, mouse.y};

    // Previous mouse position of second buffer (SDRAM_BASE)
    Position mousePrevTwo = {mouse.x, mouse.y};

    Position* mousePrevCurrent;

    if(pixel_buffer_start == FPGA_ONCHIP_BASE)
        mousePrevCurrent = &mousePrevOne;
    else if (pixel_buffer_start == SDRAM_BASE)
        mousePrevCurrent = &mousePrevTwo;
    else {
        printf("Unknown buffer");
        exit(1);
    }
	
    while (1)
    {   
        // Clears page for both buffers if page switches
        if(switchPage) {
            clear_screen();
            clear_character();
            if(switchPageCount >= 2) {
                switchPage = false;
                switchPageCount = 0;
            } else
                switchPageCount++;
        }

        // Remove previous cursor of current buffer
        removeCursor(*mousePrevCurrent);

        drawPage[currentPage]();

        // Render any event handles that occured from handlePage
        for(int i = 0; i < numOfHandles; i++) 
            handleRender[handleNum[i]]();
        

		handleNum[0] = -1;
        handleNum[1] = -1;
        handleNum[2] = -1;

        numOfHandles = 0;

        handlePage[currentPage]();
        wait_for_vsync(); // swap front and back buffers on VGA vertical sync
        pixel_buffer_start = *(pixel_ctrl_ptr + 1); // new back buffer

        int mouseXshot = mouse.x, mouseYshot = mouse.y;
        drawCursor(mouseXshot, mouseYshot);

        // Update previous mouse positions;
        (*mousePrevCurrent).x = mouseXshot;
        (*mousePrevCurrent).y = mouseYshot;

        // Update previous mouse address
        if(pixel_buffer_start == FPGA_ONCHIP_BASE)
            mousePrevCurrent = &mousePrevOne;
        else if (pixel_buffer_start == SDRAM_BASE)
            mousePrevCurrent = &mousePrevTwo;
        else {
            printf("Unknown buffer");
            exit(1);
        }
    }
}