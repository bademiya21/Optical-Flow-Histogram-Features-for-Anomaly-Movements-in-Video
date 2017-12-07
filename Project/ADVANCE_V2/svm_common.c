/************************************************************************/
/*                                                                      */
/*   svm_common.c                                                       */
/*                                                                      */
/*   Definitions and functions used in both svm_learn and svm_classify. */
/*                                                                      */
/*   Author: Thorsten Joachims                                          */
/*   Date: 02.07.04                                                     */
/*                                                                      */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved        */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include "ctype.h"
#include "svm_common.h"
#include "kernel.h"


//#ifdef __cplusplus
//extern "C" {
//#endif
//int    isnan(double);
//void   *my_malloc(size_t);
//long   maxl(long, long);
//#ifdef __cplusplus
//}
//#endif

//extern "C" int    isnan(double);
//extern "C" void   *my_malloc(size_t);
//extern "C" long   maxl(long, long);

double classify_example(MODEL *model, DOC *ex) 
	 /* classifies one example */
{
  register long i;
  register double dist;

  if((model->kernel_parm.kernel_type == LINEAR) && (model->lin_weights))
	return(classify_example_linear(model,ex));
	   
  dist=0;
  for(i=1;i<model->sv_num;i++) {  
	dist+=kernel(&model->kernel_parm,model->supvec[i],ex)*model->alpha[i];
  }
  return(dist-model->b);
}

double classify_example_linear(MODEL *model, DOC *ex) 
	 /* classifies example for linear kernel */
	 
	 /* important: the model must have the linear weight vector computed */
	 /* use: add_weight_vector_to_linear_model(&model); */


	 /* important: the feature numbers in the example to classify must */
	 /*            not be larger than the weight vector!               */
{
  double sum=0;
  SVECTOR *f;

  for(f=ex->fvec;f;f=f->next)  
	sum+=f->factor*sprod_ns(model->lin_weights,f);
  return(sum-model->b);
}


CFLOAT kernel(KERNEL_PARM *kernel_parm, DOC *a, DOC *b) 
	 /* calculate the kernel function */
{
  double sum=0;
  SVECTOR *fa,*fb;

  /* in case the constraints are sums of feature vector as represented
	 as a list of SVECTOR's with their coefficient factor in the sum,
	 take the kernel between all pairs */ 
  for(fa=a->fvec;fa;fa=fa->next) { 
	for(fb=b->fvec;fb;fb=fb->next) {
	  if(fa->kernel_id == fb->kernel_id)
	sum+=fa->factor*fb->factor*single_kernel(kernel_parm,fa,fb);
	}
  }
  return(sum);
}

CFLOAT single_kernel(KERNEL_PARM *kernel_parm, SVECTOR *a, SVECTOR *b) 
	 /* calculate the kernel function between two vectors */
{
  kernel_cache_statistic++;
  switch(kernel_parm->kernel_type) {
	case 0: /* linear */ 
			return((CFLOAT)sprod_ss(a,b)); 
	case 1: /* polynomial */
			return((CFLOAT)pow(kernel_parm->coef_lin*sprod_ss(a,b)+kernel_parm->coef_const,(double)kernel_parm->poly_degree)); 
	case 2: /* radial basis function */
			return((CFLOAT)exp(-kernel_parm->rbf_gamma*(a->twonorm_sq-2*sprod_ss(a,b)+b->twonorm_sq)));
	case 3: /* sigmoid neural net */
			return((CFLOAT)tanh(kernel_parm->coef_lin*sprod_ss(a,b)+kernel_parm->coef_const)); 
	case 4: /* custom-kernel supplied in file kernel.h*/
			return((CFLOAT)custom_kernel(kernel_parm,a,b)); 
	default: printf("Error: Unknown kernel function\n"); exit(1);
  }
}


SVECTOR *create_svector(WRD *words,FNUM n_words,char *userdefined,double factor)
{
  SVECTOR *vec;
  long    fnum,i;

  vec = (SVECTOR *)my_malloc(sizeof(SVECTOR));
  vec->n_words = n_words;

  vec->words = (FVAL*)my_malloc(sizeof(FVAL)*(n_words));
  for(i=0;i<n_words;i++) { 
	 vec->words[i]=0;
  }
  fnum=0;
  while(words[fnum].wnum) {
	 vec->words[words[fnum].wnum-1] = words[fnum].weight;
	fnum++;
  }

  vec->twonorm_sq=sprod_ss(vec,vec);

  fnum=0;
  while(userdefined[fnum]) {
	fnum++;
  }
  fnum++;
  vec->userdefined = (char *)my_malloc(sizeof(char)*(fnum));
  for(i=0;i<fnum;i++) { 
	  vec->userdefined[i]=userdefined[i];
  }
  vec->kernel_id=0;
  vec->next=NULL;
  vec->factor=factor;
  return(vec);
}

SVECTOR *create_ns_svector(const FVAL* words,FNUM n_words,char *userdefined,double factor)
{
  SVECTOR *vec;
  long i, fnum=0;
  vec = (SVECTOR *)my_malloc(sizeof(SVECTOR));
  vec->n_words = n_words;
  vec->words = (FVAL*)my_malloc(sizeof(FVAL)*(n_words));
  for(i=0;i<n_words;i++) { 
	 vec->words[i]=words[i];
  }
  vec->twonorm_sq=sprod_ss(vec,vec);

  fnum = strlen(userdefined)+1;
  vec->userdefined = (char *)my_malloc(sizeof(char)*(fnum));
  for(i=0;i<fnum;i++) { 
	  vec->userdefined[i]=userdefined[i];
  }
  vec->kernel_id=0;
  vec->next=NULL;
  vec->factor=factor;
  return(vec);
}

SVECTOR *copy_svector(SVECTOR *vec)
{
  SVECTOR *newvec=NULL;
  if(vec) {
	newvec=create_ns_svector(vec->words,vec->n_words,vec->userdefined,vec->factor);
	newvec->next=copy_svector(vec->next);
  }
  return(newvec);
}
	
void free_svector(SVECTOR *vec)
{
  if(vec) {
	free(vec->words);
	if(vec->userdefined)
	  free(vec->userdefined);
	free_svector(vec->next);
	free(vec);
  }
}

double sprod_ss(SVECTOR *a, SVECTOR *b) 
	 /* compute the inner product of two sparse vectors */
{
	register CFLOAT sum=0;
	FVAL *fa=a->words, *fb=b->words;
	FNUM n=a->n_words, i;
	for (i=0; i<n; i++) {
	   sum += fa[i]*fb[i];
	}
	return((double)sum);
}

SVECTOR* sub_ss(SVECTOR *a, SVECTOR *b) 
	 /* compute the difference a-b of two sparse vectors */
	 /* Note: SVECTOR lists are not followed, but only the first
	SVECTOR is used */
{
	SVECTOR *vec;
	FNUM n=a->n_words, i;
	FVAL *fa=a->words, *fb=b->words, *fc=(FVAL*)malloc(sizeof(FVAL)*n);
	for (i=0; i<n; i++) {
	   fc[i] = fa[i]-fb[i];
	}
	vec=create_ns_svector(fc,n,"",1.0);
	free(fc);
	return(vec);
}

SVECTOR* add_ss(SVECTOR *a, SVECTOR *b) 
	 /* compute the sum a+b of two sparse vectors */
	 /* Note: SVECTOR lists are not followed, but only the first
	SVECTOR is used */
{
	SVECTOR *vec;
	FNUM n=a->n_words, i;
	FVAL *fa=a->words, *fb=b->words, *fc=(FVAL*)malloc(sizeof(FVAL)*n);
	for (i=0; i<n; i++) {
	   fc[i] = fa[i]+fb[i];
	}
	vec=create_ns_svector(fc,n,"",1.0);
	free(fc);
	return(vec);
}

SVECTOR* add_list_ss(SVECTOR *a) 
	 /* computes the linear combination of the SVECTOR list weighted
	by the factor of each SVECTOR */
{
  SVECTOR *scaled,*oldsum,*sum,*f;
  WRD    empty[2];
	
  if(a){
	sum=smult_s(a,a->factor);
	for(f=a->next;f;f=f->next) {
	  scaled=smult_s(f,f->factor);
	  oldsum=sum;
	  sum=add_ss(sum,scaled);
	  free_svector(oldsum);
	  free_svector(scaled);
	}
	sum->factor=1.0;
  }
  else {
	empty[0].wnum=0;
	sum=create_svector(empty,0,"",1.0);
  }
  return(sum);
}

void append_svector_list(SVECTOR *a, SVECTOR *b) 
	 /* appends SVECTOR b to the end of SVECTOR a. */
{
	SVECTOR *f;
	
	for(f=a;f->next;f=f->next);  /* find end of first vector list */
	f->next=b;                   /* append the two vector lists */
}

SVECTOR* smult_s(SVECTOR *a, double factor) 
	 /* scale sparse vector a by factor */
{
	SVECTOR *vec;
	FNUM n=a->n_words, i;
	FVAL *fa=a->words, *fc=(FVAL*)malloc(sizeof(FVAL)*n);
	for (i=0; i<n; i++) {
	   fc[i] = factor*fa[i];
	}
	vec=create_ns_svector(fc,n,"",1.0);
	free(fc);
	return(vec);
}

int featvec_eq(SVECTOR *a, SVECTOR *b)
	 /* tests two sparse vectors for equality */
{
	FNUM n=a->n_words, i;
	FVAL *fa=a->words, *fb=b->words;
	if (fa == fb) return 1;
	for (i=0; i<n; i++) {
	   if (fa[i] != fb[i])
	  return 0;
	}
	return(1);
}

double model_length_s(MODEL *model, KERNEL_PARM *kernel_parm) 
	 /* compute length of weight vector */
{
  register long i,j;
  register double sum=0,alphai;
  register DOC *supveci;

  for(i=1;i<model->sv_num;i++) {  
	alphai=model->alpha[i];
	supveci=model->supvec[i];
	for(j=1;j<model->sv_num;j++) {
	  sum+=alphai*model->alpha[j]
	   *kernel(kernel_parm,supveci,model->supvec[j]);
	}
  }
  return(sqrt(sum));
}

void clear_vector_n(double *vec, long int n)
{
  register long i;
  for(i=0;i<=n;i++) vec[i]=0;
}

void add_vector_ns(double *vec_n, SVECTOR *vec_s, double faktor)
{
	FNUM n=vec_s->n_words, i;
	FVAL *fs=vec_s->words;
	for (i=0; i<n; i++) {
	   vec_n[i] += faktor*fs[i];
	}
}

double sprod_ns(double *vec_n, SVECTOR *vec_s)
{
  register double sum=0;
	FNUM n=vec_s->n_words, i;
	FVAL *fs=vec_s->words;
	for (i=0; i<n; i++) {
	   sum += vec_n[i]*fs[i];
	}
  return(sum);
}

void add_weight_vector_to_linear_model(MODEL *model)
	 /* compute weight vector in linear case and add to model */
{
  long i;
  SVECTOR *f;

  model->lin_weights=(double *)my_malloc(sizeof(double)*(model->totwords+1));
  clear_vector_n(model->lin_weights,model->totwords);
  for(i=1;i<model->sv_num;i++) {
	for(f=(model->supvec[i])->fvec;f;f=f->next)  
	  add_vector_ns(model->lin_weights,f,f->factor*model->alpha[i]);
  }
}


DOC *create_example(long docnum, long queryid, long slackid, 
			double costfactor, SVECTOR *fvec)
{
  DOC *example;
  example = (DOC *)my_malloc(sizeof(DOC));
  example->docnum=docnum;
  example->queryid=queryid;
  example->slackid=slackid;
  example->costfactor=costfactor;
  example->fvec=fvec;
  return(example);
}

void free_example(DOC *example, long deep)
{
  if(example) 
  {
	if(deep) 
	{
	  if(example->fvec)
		free_svector(example->fvec);
	}
	free(example);
  }
}

MODEL *copy_model(MODEL *model)
{
  MODEL *newmodel;
  long  i;

  newmodel=(MODEL *)my_malloc(sizeof(MODEL));
  (*newmodel)=(*model);
  newmodel->supvec = (DOC **)my_malloc(sizeof(DOC *)*model->sv_num);
  newmodel->alpha = (double *)my_malloc(sizeof(double)*model->sv_num);
  newmodel->index = NULL; /* index is not copied */
  newmodel->supvec[0] = NULL;
  newmodel->alpha[0] = 0;
  for(i=1;i<model->sv_num;i++) {
	newmodel->alpha[i]=model->alpha[i];
	newmodel->supvec[i]=create_example(model->supvec[i]->docnum,
					   model->supvec[i]->queryid,0,
					   model->supvec[i]->costfactor,
					   copy_svector(model->supvec[i]->fvec));
  }
  if(model->lin_weights) {
	newmodel->lin_weights = (double *)my_malloc(sizeof(double)*(model->totwords+1));
	for(i=0;i<model->totwords+1;i++) 
	  newmodel->lin_weights[i]=model->lin_weights[i];
  }
  return(newmodel);
}

void free_model(MODEL *model, int deep)
{
  long i;

  if(model->supvec) {
	if(deep) {
	  for(i=1;i<model->sv_num;i++) {
	free_example(model->supvec[i],1);
	  }
	}
	free(model->supvec);
  }
  if(model->alpha) free(model->alpha);
  if(model->index) free(model->index);
  if(model->lin_weights) free(model->lin_weights);
  free(model);
}


int parse_document(char *line, WRD *words, double *label,
		   long *queryid, long *slackid, double *costfactor,
		   long int *numwords, long int max_words_doc,
		   char **comment)
{
  register long wpos,pos;
  long wnum;
  double weight;
  int numread;
  char featurepair[1000],junk[1000];

  (*queryid)=0;
  (*slackid)=0;
  (*costfactor)=1;

  pos=0;
  (*comment)=NULL;
  while(line[pos] ) {      /* cut off comments */
	if((line[pos] == '#') && (!(*comment))) {
	  line[pos]=0;
	  (*comment)=&(line[pos+1]);
	}
	if(line[pos] == '\n') { /* strip the CR */
	  line[pos]=0;
	}
	pos++;
  }
  if(!(*comment)) (*comment)=&(line[pos]);
  /* printf("Comment: '%s'\n",(*comment)); */

  wpos=0;
  /* check, that line starts with target value or zero, but not with
	 feature pair */
  if(sscanf(line,"%s",featurepair) == EOF) return(0);
  pos=0;
  while((featurepair[pos] != ':') && featurepair[pos]) pos++;
  if(featurepair[pos] == ':') {
	perror ("Line must start with label or 0!!!\n"); 
	printf("LINE: %s\n",line);
	exit (1); 
  }
  /* read the target value */
  if(sscanf(line,"%lf",label) == EOF) return(0);
  pos=0;
  while(space_or_null((int)line[pos])) pos++;
  while((!space_or_null((int)line[pos])) && line[pos]) pos++;
  while(((numread=sscanf(line+pos,"%s",featurepair)) != EOF) && 
	(numread > 0) && 
	(wpos<max_words_doc)) {
	/* printf("%s\n",featurepair); */
	while(space_or_null((int)line[pos])) pos++;
	while((!space_or_null((int)line[pos])) && line[pos]) pos++;
	if(sscanf(featurepair,"qid:%ld%s",&wnum,junk)==1) {
	  /* it is the query id */
	  (*queryid)=(long)wnum;
	}
	else if(sscanf(featurepair,"sid:%ld%s",&wnum,junk)==1) {
	  /* it is the slack id */
	  if(wnum > 0) 
	(*slackid)=(long)wnum;
	  else {
	perror ("Slack-id must be greater or equal to 1!!!\n"); 
	printf("LINE: %s\n",line);
	exit (1); 
	  }
	}
	else if(sscanf(featurepair,"cost:%lf%s",&weight,junk)==1) {
	  /* it is the example-dependent cost factor */
	  (*costfactor)=(double)weight;
	}
	else if(sscanf(featurepair,"%ld:%lf%s",&wnum,&weight,junk)==2) {
	  /* it is a regular feature */
	  if(wnum<=0) { 
	perror ("Feature numbers must be larger or equal to 1!!!\n"); 
	printf("LINE: %s\n",line);
	exit (1); 
	  }
	  if((wpos>0) && ((words[wpos-1]).wnum >= wnum)) { 
	perror ("Features must be in increasing order!!!\n"); 
	printf("LINE: %s\n",line);
	exit (1); 
	  }
	  (words[wpos]).wnum=wnum;
	  (words[wpos]).weight=(FVAL)weight; 
	  wpos++;
	}
	else {
	  perror ("Cannot parse feature/value pair!!!\n"); 
	  printf("'%s' in LINE: %s\n",featurepair,line);
	  exit (1); 
	}
  }
  (words[wpos]).wnum=0;
  (*numwords)=wpos+1;
  return(1);
}

double *read_alphas(char *alphafile,long totdoc)
	 /* reads the alpha vector from a file as written by the
		write_alphas function */
{
  FILE *fl;
  double *alpha;
  long dnum;

  if ((fl = fopen (alphafile, "r")) == NULL)
  { perror (alphafile); exit (1); }

  alpha = (double *)my_malloc(sizeof(double)*totdoc);
  if(verbosity>=1) {
	printf("Reading alphas..."); fflush(stdout);
  }
  dnum=0;
  while((!feof(fl)) && fscanf(fl,"%lf\n",&alpha[dnum]) && (dnum<totdoc)) {
	dnum++;
  }
  if(dnum != totdoc)
  { perror ("\nNot enough values in alpha file!"); exit (1); }
  fclose(fl);

  if(verbosity>=1) {
	printf("done\n"); fflush(stdout);
  }

  return(alpha);
}

void nol_ll(const char *file, long int *nol, long int *wol, long int *ll) 
	 /* Grep through file and count number of lines, maximum number of
		spaces per line, and longest line. */
{
  FILE *fl;
  int ic;
  char c;
  long current_length,current_wol;

  if ((fl = fopen (file, "r")) == NULL)
  { perror (file); exit (1); }
  current_length=0;
  current_wol=0;
  (*ll)=0;
  (*nol)=1;
  (*wol)=0;
  while((ic=getc(fl)) != EOF) {
	c=(char)ic;
	current_length++;
	if(space_or_null((int)c)) {
	  current_wol++;
	}
	if(c == '\n') {
	  (*nol)++;
	  if(current_length>(*ll)) {
	(*ll)=current_length;
	  }
	  if(current_wol>(*wol)) {
	(*wol)=current_wol;
	  }
	  current_length=0;
	  current_wol=0;
	}
  }
  fclose(fl);
}

long minl(long int a, long int b)
{
  if(a<b)
	return(a);
  else
	return(b);
}

long maxl(long int a, long int b)
{
  if(a>b)
	return(a);
  else
	return(b);
}

long get_runtime(void)
{
  clock_t start;
  start = clock();
  return((long)((double)start*100.0/(double)CLOCKS_PER_SEC));
}

int isnan(double a)
{
  return(_isnan(a));
}


int space_or_null(int c) {
  if (c==0)
	return 1;
  return isspace(c);
}

void *my_malloc(size_t size)
{
  void *ptr;
  ptr=(void *)malloc(size);
  if(!ptr) { 
	perror ("Out of memory!\n"); 
	exit (1); 
  }
  return(ptr);

}

void copyright_notice(void)
{
  printf("\nCopyright: Thorsten Joachims, thorsten@joachims.org\n\n");
  printf("This software is available for non-commercial use only. It must not\n");
  printf("be modified and distributed without prior permission of the author.\n");
  printf("The author is not responsible for implications from the use of this\n");
  printf("software.\n\n");
}