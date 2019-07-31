
// compile for example:
// gcc -fPIC -shared -I/usr/include/python2.7/ -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ -lpython2.7 -O1 -march=ivybridge -mtune=ivybridge -o zbke_extension.so zbke_extension.c
// gcc -c -Q -march=native --help=target    
//



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "Python.h"
#include "arrayobject.h"



//+--- zbke parameters (citation see bottom)
#define gamma  1.2
#define mu     0.00024
#define alph   0.1
#define beta   0.000017
#define eps1   0.11
#define eps2   0.000017
#define eps3   0.0016

#define q      0.7

#define c1   (0.25/(gamma*eps2))
#define c2   (16.0*gamma*eps2)
#define c4   (alph*q)
#define c5   (gamma*eps2)

#define phi_back 1e-3

//#define het_min 0.8     // multiplicativ factor for the heterogeneity
//#define het_max 1.2





/*------------------------------------------------------------------
 *      sets pointer to c array
 * ---------------------------------------------------------------*/
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)
{
   int i,n;
   n=arrayin->dimensions[0];
   return (double *) arrayin->data;
}
    

double **ptrvector(long n)  {
    double **v;
    v=(double **)malloc((size_t) (n*sizeof(double)));
    if (!v)   {
        printf("\nIn **ptrvector. Allocation of memory for double array failed.\n");
        exit(0);  }
    return v;
} 


/*------------------------------------------------------------------
 *      sets pointer to c arrays
 * ---------------------------------------------------------------*/
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
    double **c, *a;
    unsigned int i,n,m;
    
    n=arrayin->dimensions[0];
    m=arrayin->dimensions[1];
    c=ptrvector(n);
    a=(double *) arrayin->data;  /* pointer to arrayin data as double */
    for ( i=0; i<n; i++)  {
        c[i]=a+i*m;  }
    return c;
  }


/*------------------------------------------------------------------
 *      free pointer of c arrays
 * ---------------------------------------------------------------*/
void free_Carrayptrs(double **v) { free((char*) v); }







/*------------------------------------------------------------------
 *      returns the constant zbke parameters
 * ---------------------------------------------------------------*/
static PyObject* py_zbke_param(PyObject *self)
{  return Py_BuildValue("(ddddddddd)", gamma,mu,alph,beta,eps1,eps2,eps3,q,phi_back); }




/*------------------------------------------------------------------
 *   macro function for the zbke dynamics  (C4 heterogeneity, phi background light+interaction)
 * ---------------------------------------------------------------*/
#define dxdz_dt(x,z,z_pre,C_4,phi,dt_loc,c_drei_loc)  { \
	double f1  = 1.0 - z_pre,\
	       f2  = z_pre/(eps3 + f1),\
	       uss = c1 * ( sqrt(c2*x + z_pre*z_pre - 2.0*z_pre + 1.0) -f1 );\
           x  += c_drei_loc*( phi + (mu-x)/(mu+x)*(beta+C_4*f2) + c5*uss*uss + f1*uss - x*x - x );\
           z   = z_pre + dt_loc*( 2.0*phi + f1*uss - alph*f2 ); }







/*-----------------------------------------------------------------------------------
 *                euler solver for network of zbke oscillators
 * --------------------------------------------------------------------------------*/
static PyObject* py_zbke_network(PyObject *self, PyObject *args)
{
 

    PyArrayObject *x_py, *z_py, *hetero_py,*lap_py;

    double        *x, *z, *hetero,*lap,time,dt,phi;
    
    

    if(!PyArg_ParseTuple(args, "O!O!O!O!ddd", &PyArray_Type, &x_py,   &PyArray_Type, &z_py, 
                        					  &PyArray_Type, &lap_py,  
                        					  &PyArray_Type, &hetero_py, &phi,      
                        					  &time, &dt))
    {
        printf("\nerror in input tuple parsing\n");
        return NULL;
    }
    /*
		x_py      - x component of the zbke model (1d numpy array (len:n_net), double)
		z_py      - z component of the zbke model (1d numpy array (len:n_net), double)
		lap_py    - laplace matrix of the network (1d numpy array (len:n_net*n_net, double)
		hetero_py - heterogeneity of the network unit (1d numpy array (len:n_net), double)
		phi_back  - background light intensity in the zbke model (double)
		time      - time to integrate the zbke model (double)
		dt        - timestep discretization (double)
    */


    // number of euler steps
    const long steps    = (long) (time/dt);
	const double c3_net = dt/eps1;

    long i,j,k,l;
              
    // set pointers
    x          = pyvector_to_Carrayptrs(x_py);    
    z          = pyvector_to_Carrayptrs(z_py);
    hetero     = pyvector_to_Carrayptrs(hetero_py);
    lap   	   = pyvector_to_Carrayptrs(lap_py);
    

    // network size
    int n_net = z_py->dimensions[0];


    //+--- parameter and array dimension check 
	if(   ((n_net) != hetero_py->dimensions[0]) ^ ((n_net) != x_py->dimensions[0])  )
    {   printf("\n wrong dimension in x,z components \n");   return NULL;    }

	if(   ((n_net*n_net) != lap_py->dimensions[0]) )
    {   printf("\n wrong dimension in laplace matrix \n");   return NULL;    }

	if(  (dt  < 1e-6)  ^  (dt > 2e-4) )
    {   printf("\n stepsize dt should be in [1e-6,2e-4] to ensure convergence save compution time \n");   return NULL;    }


    double  *c_4_het, // node heterogeneity
    		*inter;   // network interaction

    c_4_het = malloc( n_net * sizeof(double));
    inter   = malloc( n_net * sizeof(double));
           
    for( i=0;i<n_net;i++) {  c_4_het[i] = hetero[i]*c4; }
            

    

    
    //+--------------- euler steps ---------------------------
    for( i=0;i<steps;i++ )
    {

		// calculate the network interaction
        for( int k=0;k<n_net;k++ )
        {
             inter[k]  =  phi;
             int ix= k*n_net;
             for( int j=0; j<n_net;j++ )   inter[k] +=  lap[ix+j] *z[j];
        }
        
        
   		// call macro function with zbke dynamics
        for(k=0;k<n_net;k++)
        {
			double zpre = z[k];
            dxdz_dt( x[k], z[k], zpre, c_4_het[k], inter[k], dt,c3_net );
        }

    }//+-------------- END EULER LOOP -------------------


    free(c_4_het); free(inter);
    
    return Py_BuildValue("l",0);
}







/*-----------------------------------------------------------------------------------
 *            euler solver for network of zbke oscillators with stdp
 * --------------------------------------------------------------------------------*/
//#define phi_back_stdp  (5.3e-4)   // background light intensity


//+--- parameter of STDP model
#define a_1_stdp    1.55  // amplitude of potentiation
#define a_2_stdp    .75	  // 	,,		  depression

#define tau_1_stdp  4.  // exponential decay of potentiation
#define tau_2_stdp  12. // 			,,			depression

#define wmin_stdp   0.	// minimum coupling weight
#define wmax_stdp   1.  // maximum        ,,




#define MIN_PEAK_DISTANCE 10.0   // defines the minimum temporal distance between two consecutive peaks
#define PEAK_DELTA 0.01		// defines the minimum peak height
//#define plastic_delta 20  // in units of peak_delta




// returns the parameter of the stdp method
static PyObject* py_zbke_stdp_param(PyObject *self)
{  return Py_BuildValue("(dddd)", wmin_stdp,wmax_stdp,tau_1_stdp,tau_2_stdp); }




#define ONE_RNDMAX  (1.0/(RAND_MAX+1.))
#define TWOPI 6.2831853071795864769252867665590

/*-----------------------------------------------------------------------------------
 *                 euler method for zbke network with stdp method applied
 * --------------------------------------------------------------------------------*/
static PyObject* py_zbke_stdp(PyObject *self, PyObject *args)
{
 

    PyArrayObject *x_py, *z_py,   *zpre_py,*zppre_py,   *heterogenty_py,*adj_py,*peak_py;


    double        *x, *z,   *z_pre,*z_ppre, *heterogen,**adj,t_sys,*last_peak;


	
    const double frame_length,sigma_stdp,learnrate_stdp,noise_stdp,sim_time,dt_stdp;

  
    int wcn,pcn,zcn,zdel,zpre;
    
    const unsigned long int seed;

    if(!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!dddddl",&PyArray_Type, &x_py,           &PyArray_Type, &z_py, 
                                                      &PyArray_Type, &zpre_py,        &PyArray_Type, &zppre_py, 
                                                      &PyArray_Type, &heterogenty_py, &PyArray_Type, &adj_py,   
                                                      &PyArray_Type, &peak_py,       
                                                      &sim_time, &t_sys,&sigma_stdp, dt_stdp,
                                                      &learnrate_stdp,&noise_stdp,&seed))
    {
        printf("\n error in input tuple parsing \n");
        return NULL;
    }

    srand(seed);

	// network size
	const int n_stdp = z_py->dimensions[0];

	// global coupling strength
    const double sign_stdp        =   sigma_stdp /((double)n_stdp);

    const double c3_stdp  = dt_stdp/eps1;
    
	const long int steps   =   ( (long int)((sim_time / PEAK_DELTA ) )),
			time_peak_delta= ( (long int)((PEAK_DELTA/dt_stdp ) ));
    
	
    long int i,j,k,l;
    int wpre,ppre,pppre,ppppre;
              
    
    // bei kopplung ist delay von 1 zeiteinheit im experiment
    x          = pyvector_to_Carrayptrs(x_py);    
    z          = pyvector_to_Carrayptrs(z_py);           
    z_pre      = pyvector_to_Carrayptrs(zpre_py);        
    z_ppre     = pyvector_to_Carrayptrs(zppre_py);       
    heterogen  = pyvector_to_Carrayptrs(heterogenty_py);
    last_peak  = pyvector_to_Carrayptrs(peak_py); 
    adj        = pymatrix_to_Carrayptrs(adj_py);    

    

    //+-- define heterogeneity and allocate memory for peaks
    double *c_4_het;
    c_4_het = malloc( n_stdp*sizeof(double));
                  
    for( i=0;i<n_stdp;i++)   c_4_het[i] = heterogen[i]*c4;

	int *peaked;
	peaked = malloc( n_stdp*sizeof(int)); 

    const double noise_int = noise_stdp /sqrt(dt_stdp);



    // checks dimensions of input arrays
    if(   ((n_stdp) != z_py->dimensions[0]) ^ ((n_stdp) != x_py->dimensions[0])  )
    {   printf("\n wrong dimension in components \n");   return NULL;    }
    
    if(   ((n_stdp) != zpre_py->dimensions[0]) ^ ((n_stdp) != zppre_py->dimensions[0])  )
    {   printf("\n wrong dimension in z pre components \n");   return NULL;    }


	if(   (n_stdp != adj_py->dimensions[0] ) ^ (n_stdp != adj_py->dimensions[1]) )
    {   printf("\n dimensions of the adjacency matrix wrong \n");   return NULL;  }



    double *inter,r2,r1,exp_term,exp_term2;
    inter = malloc( n_stdp*sizeof(double));
    for( k=0;k<n_stdp;k++ ) inter[k] = 0.;



    
    // looping in seconds
	for(  int ii=0;ii<steps;ii++ )
    {
		

		// random pulse
		for( k=0;k<n_stdp;k++ )
		{
			// box-muller trafo
			r1    =  (rand()+1.) *ONE_RNDMAX;
			r2    =  (rand()+1.) *ONE_RNDMAX;
			x[k] += noise_stdp* sqrt(-log(r1))*cos( TWOPI *r2);
			
			r1    =  (rand()+1.) *ONE_RNDMAX;
			r2    =  (rand()+1.) *ONE_RNDMAX;
			z[k] += noise_stdp* sqrt(-log(r1))*cos( TWOPI *r2);
		}
		
		for(  int jj=0;jj<time_peak_delta;jj++ )
	    {
			//+-----------------------------------------------------------------
			//|                 EULER STEPS  ( all steps refer to 1 second)
			//+-----------------------------------------------------------------
			
			//+---- local dynamics
			for(k=0;k<n_stdp;k++)
			{
				double zpre = z[k];
				dxdz_dt( x[k], z[k], zpre, c_4_het[k], inter[k], dt_stdp,c3_stdp );
			}

		
			//+----- coupling interaction
			for( k=0;k<n_stdp;k++ )
			{
				double inter_tmp  =  0.;
				for( j=0; j<n_stdp;j++ )
				{
					if( k==j) continue; // self coupling neglected
					inter_tmp -=   adj[k][j] *(z[k] - z[j]);
				}
				inter[k]  =  phi_back  +  sign_stdp*inter_tmp;
			}

		}
		
		
		// clock for network time
	    t_sys += PEAK_DELTA;
	    
	    
	    for( k=0;k<n_stdp;k++ )
		{
			//-- simple peak detection
			if( (z_ppre[k] < z_pre[k]) && 	(z_pre[k] > z[k])  &&  
				(z_pre[k]>0.7)  	   &&  	(t_sys> (last_peak[k]+MIN_PEAK_DISTANCE)) )
			{
				last_peak[k] = t_sys;
				peaked[k] = 1;
			}
			else peaked[k] = 0;
			z_ppre[k] = z_pre[k];
		    z_pre[k]  = z[k];
		}


		for( i=0;i<n_stdp;i++ )
		{

			//--- synaptic plasticity following Hebbian learning
			if(  peaked[i]==1 )
			{
				for( j=i+1; j<n_stdp;j++ )
				{
					 //+--- one weight
					 double peak_diff = last_peak[i] - last_peak[j];
                     
					 if( peak_diff >= 0 )
					 {
						 exp_term  =  learnrate_stdp*a_1_stdp*exp( -peak_diff / tau_1_stdp) ;
                         exp_term2 = -learnrate_stdp*a_2_stdp*exp( -peak_diff / tau_2_stdp) ; 
					 }
					 else
					 {
						 exp_term  = -learnrate_stdp*a_2_stdp*exp(  peak_diff / tau_2_stdp) ; 
                         exp_term2 =  learnrate_stdp*a_1_stdp*exp(  peak_diff / tau_1_stdp) ;
					 }

					 adj[i][j]  +=  exp_term;
					 adj[j][i]  +=  exp_term2;

                     if( adj[i][j] <  wmin_stdp )  adj[i][j] = wmin_stdp;
                     if( adj[i][j] >  wmax_stdp )  adj[i][j] = wmax_stdp;

                     if( adj[j][i] <  wmin_stdp )  adj[j][i] = wmin_stdp;
                     if( adj[j][i] >  wmax_stdp )  adj[j][i] = wmax_stdp;

			 	 }
			}
		}

    }//+-------------- END EULER LOOP -------------------

    free_Carrayptrs(adj);
    free(c_4_het); 
    free(inter);


    
    return Py_BuildValue("d",t_sys);
}









/*-----------------------------------------------------------------------------------
 *             nullclines of the zbke model
 * --------------------------------------------------------------------------------*/
static PyObject* py_zbke_nullclines(PyObject *self, PyObject *args)
{
 
    double        xpy,zpy,phi,het;
    
 
    if(!PyArg_ParseTuple(args, "dddd",&xpy,&zpy,&phi,&het))
    {
        printf("\n error in input tuple parsing \n");
        return NULL;
    }

    
    double f1  = 1.0 - zpy;
	double f2  = zpy/(eps3 + f1);
	double uss = c1 * ( sqrt(c2*xpy + zpy*zpy - 2.0*zpy + 1.0) -f1 );
	
	double nx = (1./eps1)*( phi + (mu-xpy)/(mu+xpy)*(beta + het*c4*f2) + c5*uss*uss + f1*uss - xpy*xpy - xpy );
    double nz = 2.0*phi + f1*uss - alph*f2 ; 


    return Py_BuildValue("(dd)",nx,nz);
}





/*------------------------------------------------------
 * Bind Python function names to our C functions
 *-----------------------------------------------------*/
static PyMethodDef myModule_methods[] = {
  {"zbke_network",             py_zbke_network,  METH_VARARGS},
  {"zbke_param",               py_zbke_param,    METH_VARARGS},
  {"zbke_stdp_param",		   py_zbke_stdp_param, METH_VARARGS},
  {"zbke_nullclines",          py_zbke_nullclines,    METH_VARARGS},
  {"zbke_stdp",                py_zbke_stdp,    METH_VARARGS},
  {NULL,NULL}
};

/*----------------------------------------------------------------------------------------
 * Python calls this to let us initialize our module
 *------------------------------------------------------------------------------------- */
void initzbke_extension()
{
  (void) Py_InitModule("zbke_extension", myModule_methods);
  import_array();
}


/* 
	The ZBKE model is presented in the following publication:

	Anatol M. Zhabotinsky, Frank Buchholtz, Anatol B. Kiyatkin, and Irving R. Epstein
	"Oscillations and waves in metal-ion-catalyzed bromate oscillating reactions in highly oxidized states"
	J. Phys. Chem. 1993,  97, 29, 7578-7584
	https://pubs.acs.org/doi/abs/10.1021/j100131a030?journalCode=jpchax
	10.1021/j100131a030
*/