#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// number of iterations
#ifndef M
	#define M 100
#endif

#define SPACE_SIZE 1000

// the type used to represent a triple of doubles
typedef struct {
	double x, y, z;
} triple;

// the types used to model position, speed and forces
typedef triple position;
typedef triple velocity;
typedef triple force;
typedef triple impulse;

// the type used to model one body
typedef struct {
	double m;		// the mass of the body
	position pos;		// the position in space
	velocity v;		// the velocity of the body
} body;

// ----- utility functions ------
double rand_val(double min, double max) {
	return (rand() / (double) RAND_MAX) * (max - min) + min;
}

triple triple_zero() {
	return (position) {0.0, 0.0, 0.0};
}

triple triple_rand() {
	return (position) {
		rand_val(-SPACE_SIZE,SPACE_SIZE),
		rand_val(-SPACE_SIZE,SPACE_SIZE),
		rand_val(-SPACE_SIZE,SPACE_SIZE)
	};
}

void triple_print(triple t) {
	printf("(%f,%f,%f)", t.x, t.y, t.z);
}

// some operators
#define eps 0.0001
#define abs(V) (((V)<0)?-(V):(V))

#define ADD(T1,T2) 		(triple) { (T1).x + (T2).x, (T1).y + (T2).y, (T1).z + (T2).z }
#define SUB(T1,T2) 		(triple) { (T1).x - (T2).x, (T1).y - (T2).y, (T1).z - (T2).z }

#define MULS(T,S) 		(triple) { (T).x * (S), (T).y * (S), (T).z * (S) }
#define DIVS(T,S) 		(triple) { (T).x / (S), (T).y / (S), (T).z / (S) }

#define EQ(T1,T2) 		(abs((T1).x-(T2).x) < eps && abs((T1).y-(T2).y) < eps && abs((T1).z-(T2).z) < eps)

#define ABS(T)			sqrt((T).x*(T).x + (T).y*(T).y + (T).z*(T).z)

int main(int argc, char** argv) {

	if(argc != 3) {
		printf("usage: %s <num_bodies> <seed>\n", argv[0]);
		return EXIT_FAILURE;
	}

	const long long numBodies = atol(argv[1]);
	const unsigned seed = atol(argv[2]);

	// the list of bodies
	body* bodies = malloc(sizeof(body) * numBodies);

	// the forces effecting the particless
	force* forces = malloc(sizeof(force) * numBodies);

	srand(seed);

	// distribute bodies in space (randomly)
	for(int i=0; i<numBodies; i++) {
		bodies[i].m = 1.0;
		bodies[i].pos = triple_rand();
		//bodies[i].pos = (position) { 0, -10 + 20*(i/2), -10 + 20*(i%2) }; // for debugging!
		bodies[i].v   = triple_zero();
	}

	// run simulation for M steps
	for(int i=0; i<M; i++) {
		
		// set forces to zero
		for(int j=0; j<numBodies; j++) {
			forces[j] = triple_zero();
		}

		// compute forces for each body (very naive)
		for(int j=0; j<numBodies; j++) {
			for(int k=0; k<numBodies; k++) {
				if(j!=k) {
					// comput distance vector
					triple dist = SUB(bodies[k].pos, bodies[j].pos);

					// compute absolute distance
					double r = ABS(dist);
				
					// compute strength of force (G = 1 (who cares))
					// F = G * (m1 * m2) / r^2
					double f = (bodies[j].m * bodies[k].m) / (r*r);

					// compute current contribution to force
					double s = f / r;
					force cur = MULS(dist,s);

					// accumulate force
					forces[j] = ADD(forces[j], cur);
				}
			}
		}

		// apply forces
		for(int j=0; j<numBodies; j++) {
			// update speed
			// F = m * a
			// a = F / m
			// v' = v + a
			bodies[j].v = ADD(bodies[j].v, DIVS(forces[j], bodies[j].m));

			// update position
			// pos = pos + v * dt		// dt = 1
			bodies[j].pos = ADD(bodies[j].pos, bodies[j].v);
		}

	}
	
	// debug print of final positions and speed
	for(int i=0; i<numBodies; i++) {
		printf("%2d - ", i); 
		triple_print(bodies[i].pos);
		printf(" - ");
		triple_print(bodies[i].v);
		printf("\n");
	}
	printf("\n");

	// check result (impulse has to be zero)
	impulse sum = triple_zero();
	for(int i=0; i<numBodies; i++) {
		// impulse = m * v
		sum = ADD(sum, MULS(bodies[i].v,bodies[i].m));
	}
	int success = EQ(sum, triple_zero());
	printf("Verification: %s\n", ((success)?"OK":"ERR"));
	if (!success) {
		triple_print(sum); printf(" should be (0,0,0)\n");
		return EXIT_FAILURE;
	}

	// cleanup
	free(bodies);
	free(forces);

	return EXIT_SUCCESS;
}
