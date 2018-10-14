



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "flame_globals.h"
//#include "CustomVisualisation.h"

void initGlobalsController();


//
//extern void set_TISSUE_DAMAGE_PROB(float* prob);
//extern void set_QUIESCENT_MIGRATION_SCALE(float* scale);
//extern void set_REPAIR_RANGE(float* range);
//extern void set_DAMAGE_DETECTION_RANGE(float* range);
//extern void set_REPAIR_RATE(int* num_per_frame);
//
//extern const float * get_TISSUE_DAMAGE_PROB();
//extern const float * get_QUIESCENT_MIGRATION_SCALE();
//extern const float * get_REPAIR_RANGE();
//extern const float * get_DAMAGE_DETECTION_RANGE();
//extern const int * get_REPAIR_RATE();

float tissueDamageProb = 0;
float quiescentMigrationScale = 0;
int repairRate = 0;
float damageDetectionRange = 0;
float repairRange = 0;


void initGlobalsController()
{
    tissueDamageProb = *get_TISSUE_DAMAGE_PROB();
    quiescentMigrationScale = *get_QUIESCENT_MIGRATION_SCALE();
    repairRate = *get_REPAIR_RATE();
    damageDetectionRange = *get_DAMAGE_DETECTION_RANGE();
    repairRange = *get_REPAIR_RANGE();
}
























