//
// Created by cwels on 13/10/2018.
//

#ifndef __FLAME_GLOBALS_H
#define __FLAME_GLOBALS_H

void initGlobalsController();


extern void set_TISSUE_DAMAGE_PROB(float* prob);
extern void set_QUIESCENT_MIGRATION_SCALE(float* scale);
extern void set_REPAIR_RANGE(float* range);
extern void set_DAMAGE_DETECTION_RANGE(float* range);
extern void set_REPAIR_RATE(int* num_per_frame);

extern const float * get_TISSUE_DAMAGE_PROB();
extern const float * get_QUIESCENT_MIGRATION_SCALE();
extern const float * get_REPAIR_RANGE();
extern const float * get_DAMAGE_DETECTION_RANGE();
extern const int * get_REPAIR_RATE();


















#endif //FIRSTFLAMEPROJECT_FLAME_GLOBALS_H
