??*
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??$
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
instance_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameinstance_normalization/gamma
?
0instance_normalization/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization/gamma*
_output_shapes
:*
dtype0
?
instance_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameinstance_normalization/beta
?
/instance_normalization/beta/Read/ReadVariableOpReadVariableOpinstance_normalization/beta*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
?
instance_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name instance_normalization_1/gamma
?
2instance_normalization_1/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_1/gamma*
_output_shapes
: *
dtype0
?
instance_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameinstance_normalization_1/beta
?
1instance_normalization_1/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_1/beta*
_output_shapes
: *
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: 0*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:0*
dtype0
?
instance_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*/
shared_name instance_normalization_2/gamma
?
2instance_normalization_2/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_2/gamma*
_output_shapes
:0*
dtype0
?
instance_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*.
shared_nameinstance_normalization_2/beta
?
1instance_normalization_2/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_2/beta*
_output_shapes
:0*
dtype0
?
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*(
shared_nameconv2d_transpose/kernel
?
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*&
_output_shapes
: 0*
dtype0
?
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
: *
dtype0
?
instance_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name instance_normalization_3/gamma
?
2instance_normalization_3/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_3/gamma*
_output_shapes
: *
dtype0
?
instance_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameinstance_normalization_3/beta
?
1instance_normalization_3/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_3/beta*
_output_shapes
: *
dtype0
?
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_1/kernel
?
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
:*
dtype0
?
instance_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name instance_normalization_4/gamma
?
2instance_normalization_4/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_4/gamma*
_output_shapes
:*
dtype0
?
instance_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameinstance_normalization_4/beta
?
1instance_normalization_4/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_4/beta*
_output_shapes
:*
dtype0
?
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_13/kernel
}
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*&
_output_shapes
:*
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
:*
dtype0
?
instance_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name instance_normalization_5/gamma
?
2instance_normalization_5/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_5/gamma*
_output_shapes
:*
dtype0
?
instance_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameinstance_normalization_5/beta
?
1instance_normalization_5/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_5/beta*
_output_shapes
:*
dtype0
?
residual_block/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*/
shared_name residual_block/conv2d_3/kernel
?
2residual_block/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpresidual_block/conv2d_3/kernel*&
_output_shapes
:00*
dtype0
?
residual_block/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*-
shared_nameresidual_block/conv2d_3/bias
?
0residual_block/conv2d_3/bias/Read/ReadVariableOpReadVariableOpresidual_block/conv2d_3/bias*
_output_shapes
:0*
dtype0
?
residual_block/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*/
shared_name residual_block/conv2d_4/kernel
?
2residual_block/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpresidual_block/conv2d_4/kernel*&
_output_shapes
:00*
dtype0
?
residual_block/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*-
shared_nameresidual_block/conv2d_4/bias
?
0residual_block/conv2d_4/bias/Read/ReadVariableOpReadVariableOpresidual_block/conv2d_4/bias*
_output_shapes
:0*
dtype0
?
 residual_block_1/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*1
shared_name" residual_block_1/conv2d_5/kernel
?
4residual_block_1/conv2d_5/kernel/Read/ReadVariableOpReadVariableOp residual_block_1/conv2d_5/kernel*&
_output_shapes
:00*
dtype0
?
residual_block_1/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*/
shared_name residual_block_1/conv2d_5/bias
?
2residual_block_1/conv2d_5/bias/Read/ReadVariableOpReadVariableOpresidual_block_1/conv2d_5/bias*
_output_shapes
:0*
dtype0
?
 residual_block_1/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*1
shared_name" residual_block_1/conv2d_6/kernel
?
4residual_block_1/conv2d_6/kernel/Read/ReadVariableOpReadVariableOp residual_block_1/conv2d_6/kernel*&
_output_shapes
:00*
dtype0
?
residual_block_1/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*/
shared_name residual_block_1/conv2d_6/bias
?
2residual_block_1/conv2d_6/bias/Read/ReadVariableOpReadVariableOpresidual_block_1/conv2d_6/bias*
_output_shapes
:0*
dtype0
?
 residual_block_2/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*1
shared_name" residual_block_2/conv2d_7/kernel
?
4residual_block_2/conv2d_7/kernel/Read/ReadVariableOpReadVariableOp residual_block_2/conv2d_7/kernel*&
_output_shapes
:00*
dtype0
?
residual_block_2/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*/
shared_name residual_block_2/conv2d_7/bias
?
2residual_block_2/conv2d_7/bias/Read/ReadVariableOpReadVariableOpresidual_block_2/conv2d_7/bias*
_output_shapes
:0*
dtype0
?
 residual_block_2/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*1
shared_name" residual_block_2/conv2d_8/kernel
?
4residual_block_2/conv2d_8/kernel/Read/ReadVariableOpReadVariableOp residual_block_2/conv2d_8/kernel*&
_output_shapes
:00*
dtype0
?
residual_block_2/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*/
shared_name residual_block_2/conv2d_8/bias
?
2residual_block_2/conv2d_8/bias/Read/ReadVariableOpReadVariableOpresidual_block_2/conv2d_8/bias*
_output_shapes
:0*
dtype0
?
 residual_block_3/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*1
shared_name" residual_block_3/conv2d_9/kernel
?
4residual_block_3/conv2d_9/kernel/Read/ReadVariableOpReadVariableOp residual_block_3/conv2d_9/kernel*&
_output_shapes
:00*
dtype0
?
residual_block_3/conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*/
shared_name residual_block_3/conv2d_9/bias
?
2residual_block_3/conv2d_9/bias/Read/ReadVariableOpReadVariableOpresidual_block_3/conv2d_9/bias*
_output_shapes
:0*
dtype0
?
!residual_block_3/conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*2
shared_name#!residual_block_3/conv2d_10/kernel
?
5residual_block_3/conv2d_10/kernel/Read/ReadVariableOpReadVariableOp!residual_block_3/conv2d_10/kernel*&
_output_shapes
:00*
dtype0
?
residual_block_3/conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*0
shared_name!residual_block_3/conv2d_10/bias
?
3residual_block_3/conv2d_10/bias/Read/ReadVariableOpReadVariableOpresidual_block_3/conv2d_10/bias*
_output_shapes
:0*
dtype0
?
!residual_block_4/conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*2
shared_name#!residual_block_4/conv2d_11/kernel
?
5residual_block_4/conv2d_11/kernel/Read/ReadVariableOpReadVariableOp!residual_block_4/conv2d_11/kernel*&
_output_shapes
:00*
dtype0
?
residual_block_4/conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*0
shared_name!residual_block_4/conv2d_11/bias
?
3residual_block_4/conv2d_11/bias/Read/ReadVariableOpReadVariableOpresidual_block_4/conv2d_11/bias*
_output_shapes
:0*
dtype0
?
!residual_block_4/conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*2
shared_name#!residual_block_4/conv2d_12/kernel
?
5residual_block_4/conv2d_12/kernel/Read/ReadVariableOpReadVariableOp!residual_block_4/conv2d_12/kernel*&
_output_shapes
:00*
dtype0
?
residual_block_4/conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*0
shared_name!residual_block_4/conv2d_12/bias
?
3residual_block_4/conv2d_12/bias/Read/ReadVariableOpReadVariableOpresidual_block_4/conv2d_12/bias*
_output_shapes
:0*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
layer_with_weights-11
layer-14
layer_with_weights-12
layer-15
layer-16
layer_with_weights-13
layer-17
layer_with_weights-14
layer-18
layer-19
layer_with_weights-15
layer-20
layer_with_weights-16
layer-21
layer-22
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
g
	#gamma
$beta
%trainable_variables
&	variables
'regularization_losses
(	keras_api
R
)trainable_variables
*	variables
+regularization_losses
,	keras_api
h

-kernel
.bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
g
	3gamma
4beta
5trainable_variables
6	variables
7regularization_losses
8	keras_api
R
9trainable_variables
:	variables
;regularization_losses
<	keras_api
h

=kernel
>bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
g
	Cgamma
Dbeta
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
R
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
h
	Mconv1
	Nconv2
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
h
	Sconv1
	Tconv2
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
h
	Yconv1
	Zconv2
[trainable_variables
\	variables
]regularization_losses
^	keras_api
h
	_conv1
	`conv2
atrainable_variables
b	variables
cregularization_losses
d	keras_api
h
	econv1
	fconv2
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
h

kkernel
lbias
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
g
	qgamma
rbeta
strainable_variables
t	variables
uregularization_losses
v	keras_api
R
wtrainable_variables
x	variables
yregularization_losses
z	keras_api
i

{kernel
|bias
}trainable_variables
~	variables
regularization_losses
?	keras_api
m

?gamma
	?beta
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
m

?gamma
	?beta
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
0
1
#2
$3
-4
.5
36
47
=8
>9
C10
D11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
k32
l33
q34
r35
{36
|37
?38
?39
?40
?41
?42
?43
?
0
1
#2
$3
-4
.5
36
47
=8
>9
C10
D11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
k32
l33
q34
r35
{36
|37
?38
?39
?40
?41
?42
?43
 
?
?layers
 ?layer_regularization_losses
trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
	variables
regularization_losses
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
?layers
 ?layer_regularization_losses
trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
 	variables
!regularization_losses
ge
VARIABLE_VALUEinstance_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEinstance_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
?
?layers
 ?layer_regularization_losses
%trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
&	variables
'regularization_losses
 
 
 
?
?layers
 ?layer_regularization_losses
)trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
*	variables
+regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 
?
?layers
 ?layer_regularization_losses
/trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
0	variables
1regularization_losses
ig
VARIABLE_VALUEinstance_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEinstance_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE

30
41

30
41
 
?
?layers
 ?layer_regularization_losses
5trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
6	variables
7regularization_losses
 
 
 
?
?layers
 ?layer_regularization_losses
9trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
:	variables
;regularization_losses
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

=0
>1
 
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
@	variables
Aregularization_losses
ig
VARIABLE_VALUEinstance_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEinstance_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

C0
D1
 
?
?layers
 ?layer_regularization_losses
Etrainable_variables
?non_trainable_variables
?metrics
?layer_metrics
F	variables
Gregularization_losses
 
 
 
?
?layers
 ?layer_regularization_losses
Itrainable_variables
?non_trainable_variables
?metrics
?layer_metrics
J	variables
Kregularization_losses
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
 
?0
?1
?2
?3
 
?0
?1
?2
?3
 
?
?layers
 ?layer_regularization_losses
Otrainable_variables
?non_trainable_variables
?metrics
?layer_metrics
P	variables
Qregularization_losses
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
 
?0
?1
?2
?3
 
?0
?1
?2
?3
 
?
?layers
 ?layer_regularization_losses
Utrainable_variables
?non_trainable_variables
?metrics
?layer_metrics
V	variables
Wregularization_losses
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
 
?0
?1
?2
?3
 
?0
?1
?2
?3
 
?
?layers
 ?layer_regularization_losses
[trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
\	variables
]regularization_losses
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
 
?0
?1
?2
?3
 
?0
?1
?2
?3
 
?
?layers
 ?layer_regularization_losses
atrainable_variables
?non_trainable_variables
?metrics
?layer_metrics
b	variables
cregularization_losses
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
 
?0
?1
?2
?3
 
?0
?1
?2
?3
 
?
?layers
 ?layer_regularization_losses
gtrainable_variables
?non_trainable_variables
?metrics
?layer_metrics
h	variables
iregularization_losses
db
VARIABLE_VALUEconv2d_transpose/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_transpose/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

k0
l1

k0
l1
 
?
?layers
 ?layer_regularization_losses
mtrainable_variables
?non_trainable_variables
?metrics
?layer_metrics
n	variables
oregularization_losses
jh
VARIABLE_VALUEinstance_normalization_3/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEinstance_normalization_3/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE

q0
r1

q0
r1
 
?
?layers
 ?layer_regularization_losses
strainable_variables
?non_trainable_variables
?metrics
?layer_metrics
t	variables
uregularization_losses
 
 
 
?
?layers
 ?layer_regularization_losses
wtrainable_variables
?non_trainable_variables
?metrics
?layer_metrics
x	variables
yregularization_losses
fd
VARIABLE_VALUEconv2d_transpose_1/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_1/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

{0
|1

{0
|1
 
?
?layers
 ?layer_regularization_losses
}trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
~	variables
regularization_losses
jh
VARIABLE_VALUEinstance_normalization_4/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEinstance_normalization_4/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
 
 
 
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
][
VARIABLE_VALUEconv2d_13/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_13/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
jh
VARIABLE_VALUEinstance_normalization_5/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEinstance_normalization_5/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
 
 
 
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
ec
VARIABLE_VALUEresidual_block/conv2d_3/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEresidual_block/conv2d_3/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresidual_block/conv2d_4/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEresidual_block/conv2d_4/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE residual_block_1/conv2d_5/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresidual_block_1/conv2d_5/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE residual_block_1/conv2d_6/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresidual_block_1/conv2d_6/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE residual_block_2/conv2d_7/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresidual_block_2/conv2d_7/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE residual_block_2/conv2d_8/kernel1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresidual_block_2/conv2d_8/bias1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE residual_block_3/conv2d_9/kernel1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresidual_block_3/conv2d_9/bias1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!residual_block_3/conv2d_10/kernel1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresidual_block_3/conv2d_10/bias1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!residual_block_4/conv2d_11/kernel1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresidual_block_4/conv2d_11/bias1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!residual_block_4/conv2d_12/kernel1trainable_variables/30/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresidual_block_4/conv2d_12/bias1trainable_variables/31/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1

?0
?1
 
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses

?0
?1

?0
?1
 
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses

M0
N1
 
 
 
 

?0
?1

?0
?1
 
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses

?0
?1

?0
?1
 
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses

S0
T1
 
 
 
 

?0
?1

?0
?1
 
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses

?0
?1

?0
?1
 
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses

Y0
Z1
 
 
 
 

?0
?1

?0
?1
 
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses

?0
?1

?0
?1
 
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses

_0
`1
 
 
 
 

?0
?1

?0
?1
 
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses

?0
?1

?0
?1
 
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses

e0
f1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_conv2d_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelconv2d/biasinstance_normalization/gammainstance_normalization/betaconv2d_1/kernelconv2d_1/biasinstance_normalization_1/gammainstance_normalization_1/betaconv2d_2/kernelconv2d_2/biasinstance_normalization_2/gammainstance_normalization_2/betaresidual_block/conv2d_3/kernelresidual_block/conv2d_3/biasresidual_block/conv2d_4/kernelresidual_block/conv2d_4/bias residual_block_1/conv2d_5/kernelresidual_block_1/conv2d_5/bias residual_block_1/conv2d_6/kernelresidual_block_1/conv2d_6/bias residual_block_2/conv2d_7/kernelresidual_block_2/conv2d_7/bias residual_block_2/conv2d_8/kernelresidual_block_2/conv2d_8/bias residual_block_3/conv2d_9/kernelresidual_block_3/conv2d_9/bias!residual_block_3/conv2d_10/kernelresidual_block_3/conv2d_10/bias!residual_block_4/conv2d_11/kernelresidual_block_4/conv2d_11/bias!residual_block_4/conv2d_12/kernelresidual_block_4/conv2d_12/biasconv2d_transpose/kernelconv2d_transpose/biasinstance_normalization_3/gammainstance_normalization_3/betaconv2d_transpose_1/kernelconv2d_transpose_1/biasinstance_normalization_4/gammainstance_normalization_4/betaconv2d_13/kernelconv2d_13/biasinstance_normalization_5/gammainstance_normalization_5/beta*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_6353
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp0instance_normalization/gamma/Read/ReadVariableOp/instance_normalization/beta/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp2instance_normalization_1/gamma/Read/ReadVariableOp1instance_normalization_1/beta/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp2instance_normalization_2/gamma/Read/ReadVariableOp1instance_normalization_2/beta/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp2instance_normalization_3/gamma/Read/ReadVariableOp1instance_normalization_3/beta/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp2instance_normalization_4/gamma/Read/ReadVariableOp1instance_normalization_4/beta/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp2instance_normalization_5/gamma/Read/ReadVariableOp1instance_normalization_5/beta/Read/ReadVariableOp2residual_block/conv2d_3/kernel/Read/ReadVariableOp0residual_block/conv2d_3/bias/Read/ReadVariableOp2residual_block/conv2d_4/kernel/Read/ReadVariableOp0residual_block/conv2d_4/bias/Read/ReadVariableOp4residual_block_1/conv2d_5/kernel/Read/ReadVariableOp2residual_block_1/conv2d_5/bias/Read/ReadVariableOp4residual_block_1/conv2d_6/kernel/Read/ReadVariableOp2residual_block_1/conv2d_6/bias/Read/ReadVariableOp4residual_block_2/conv2d_7/kernel/Read/ReadVariableOp2residual_block_2/conv2d_7/bias/Read/ReadVariableOp4residual_block_2/conv2d_8/kernel/Read/ReadVariableOp2residual_block_2/conv2d_8/bias/Read/ReadVariableOp4residual_block_3/conv2d_9/kernel/Read/ReadVariableOp2residual_block_3/conv2d_9/bias/Read/ReadVariableOp5residual_block_3/conv2d_10/kernel/Read/ReadVariableOp3residual_block_3/conv2d_10/bias/Read/ReadVariableOp5residual_block_4/conv2d_11/kernel/Read/ReadVariableOp3residual_block_4/conv2d_11/bias/Read/ReadVariableOp5residual_block_4/conv2d_12/kernel/Read/ReadVariableOp3residual_block_4/conv2d_12/bias/Read/ReadVariableOpConst*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_8099
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasinstance_normalization/gammainstance_normalization/betaconv2d_1/kernelconv2d_1/biasinstance_normalization_1/gammainstance_normalization_1/betaconv2d_2/kernelconv2d_2/biasinstance_normalization_2/gammainstance_normalization_2/betaconv2d_transpose/kernelconv2d_transpose/biasinstance_normalization_3/gammainstance_normalization_3/betaconv2d_transpose_1/kernelconv2d_transpose_1/biasinstance_normalization_4/gammainstance_normalization_4/betaconv2d_13/kernelconv2d_13/biasinstance_normalization_5/gammainstance_normalization_5/betaresidual_block/conv2d_3/kernelresidual_block/conv2d_3/biasresidual_block/conv2d_4/kernelresidual_block/conv2d_4/bias residual_block_1/conv2d_5/kernelresidual_block_1/conv2d_5/bias residual_block_1/conv2d_6/kernelresidual_block_1/conv2d_6/bias residual_block_2/conv2d_7/kernelresidual_block_2/conv2d_7/bias residual_block_2/conv2d_8/kernelresidual_block_2/conv2d_8/bias residual_block_3/conv2d_9/kernelresidual_block_3/conv2d_9/bias!residual_block_3/conv2d_10/kernelresidual_block_3/conv2d_10/bias!residual_block_4/conv2d_11/kernelresidual_block_4/conv2d_11/bias!residual_block_4/conv2d_12/kernelresidual_block_4/conv2d_12/bias*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_8241??!
?
?
C__inference_conv2d_11_layer_call_and_return_conditional_losses_4797

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
'__inference_StyleNet_layer_call_fn_7208

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7: 0
	unknown_8:0
	unknown_9:0

unknown_10:0$

unknown_11:00

unknown_12:0$

unknown_13:00

unknown_14:0$

unknown_15:00

unknown_16:0$

unknown_17:00

unknown_18:0$

unknown_19:00

unknown_20:0$

unknown_21:00

unknown_22:0$

unknown_23:00

unknown_24:0$

unknown_25:00

unknown_26:0$

unknown_27:00

unknown_28:0$

unknown_29:00

unknown_30:0$

unknown_31: 0

unknown_32: 

unknown_33: 

unknown_34: $

unknown_35: 

unknown_36:

unknown_37:

unknown_38:$

unknown_39:

unknown_40:

unknown_41:

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_StyleNet_layer_call_and_return_conditional_losses_54072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_4501

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
C__inference_conv2d_11_layer_call_and_return_conditional_losses_7916

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
b
F__inference_activation_3_layer_call_and_return_conditional_losses_7601

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
B__inference_conv2d_9_layer_call_and_return_conditional_losses_4723

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
??
?
B__inference_StyleNet_layer_call_and_return_conditional_losses_5844

inputs%
conv2d_5732:
conv2d_5734:)
instance_normalization_5737:)
instance_normalization_5739:'
conv2d_1_5743: 
conv2d_1_5745: +
instance_normalization_1_5748: +
instance_normalization_1_5750: '
conv2d_2_5754: 0
conv2d_2_5756:0+
instance_normalization_2_5759:0+
instance_normalization_2_5761:0-
residual_block_5765:00!
residual_block_5767:0-
residual_block_5769:00!
residual_block_5771:0/
residual_block_1_5774:00#
residual_block_1_5776:0/
residual_block_1_5778:00#
residual_block_1_5780:0/
residual_block_2_5783:00#
residual_block_2_5785:0/
residual_block_2_5787:00#
residual_block_2_5789:0/
residual_block_3_5792:00#
residual_block_3_5794:0/
residual_block_3_5796:00#
residual_block_3_5798:0/
residual_block_4_5801:00#
residual_block_4_5803:0/
residual_block_4_5805:00#
residual_block_4_5807:0/
conv2d_transpose_5810: 0#
conv2d_transpose_5812: +
instance_normalization_3_5815: +
instance_normalization_3_5817: 1
conv2d_transpose_1_5821: %
conv2d_transpose_1_5823:+
instance_normalization_4_5826:+
instance_normalization_4_5828:(
conv2d_13_5832:
conv2d_13_5834:+
instance_normalization_5_5837:+
instance_normalization_5_5839:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?.instance_normalization/StatefulPartitionedCall?0instance_normalization_1/StatefulPartitionedCall?0instance_normalization_2/StatefulPartitionedCall?0instance_normalization_3/StatefulPartitionedCall?0instance_normalization_4/StatefulPartitionedCall?0instance_normalization_5/StatefulPartitionedCall?&residual_block/StatefulPartitionedCall?(residual_block_1/StatefulPartitionedCall?(residual_block_2/StatefulPartitionedCall?(residual_block_3/StatefulPartitionedCall?(residual_block_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5732conv2d_5734*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_49612 
conv2d/StatefulPartitionedCall?
.instance_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0instance_normalization_5737instance_normalization_5739*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_instance_normalization_layer_call_and_return_conditional_losses_501020
.instance_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall7instance_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_50212
activation/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_5743conv2d_1_5745*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_50332"
 conv2d_1/StatefulPartitionedCall?
0instance_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0instance_normalization_1_5748instance_normalization_1_5750*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_1_layer_call_and_return_conditional_losses_508222
0instance_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall9instance_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_50932
activation_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_5754conv2d_2_5756*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_51052"
 conv2d_2/StatefulPartitionedCall?
0instance_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0instance_normalization_2_5759instance_normalization_2_5761*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_2_layer_call_and_return_conditional_losses_515422
0instance_normalization_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall9instance_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_51652
activation_2/PartitionedCall?
&residual_block/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0residual_block_5765residual_block_5767residual_block_5769residual_block_5771*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_residual_block_layer_call_and_return_conditional_losses_45262(
&residual_block/StatefulPartitionedCall?
(residual_block_1/StatefulPartitionedCallStatefulPartitionedCall/residual_block/StatefulPartitionedCall:output:0residual_block_1_5774residual_block_1_5776residual_block_1_5778residual_block_1_5780*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_1_layer_call_and_return_conditional_losses_46002*
(residual_block_1/StatefulPartitionedCall?
(residual_block_2/StatefulPartitionedCallStatefulPartitionedCall1residual_block_1/StatefulPartitionedCall:output:0residual_block_2_5783residual_block_2_5785residual_block_2_5787residual_block_2_5789*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_2_layer_call_and_return_conditional_losses_46742*
(residual_block_2/StatefulPartitionedCall?
(residual_block_3/StatefulPartitionedCallStatefulPartitionedCall1residual_block_2/StatefulPartitionedCall:output:0residual_block_3_5792residual_block_3_5794residual_block_3_5796residual_block_3_5798*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_3_layer_call_and_return_conditional_losses_47482*
(residual_block_3/StatefulPartitionedCall?
(residual_block_4/StatefulPartitionedCallStatefulPartitionedCall1residual_block_3/StatefulPartitionedCall:output:0residual_block_4_5801residual_block_4_5803residual_block_4_5805residual_block_4_5807*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_4_layer_call_and_return_conditional_losses_48222*
(residual_block_4/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall1residual_block_4/StatefulPartitionedCall:output:0conv2d_transpose_5810conv2d_transpose_5812*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_48902*
(conv2d_transpose/StatefulPartitionedCall?
0instance_normalization_3/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0instance_normalization_3_5815instance_normalization_3_5817*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_3_layer_call_and_return_conditional_losses_526022
0instance_normalization_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall9instance_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_52712
activation_3/PartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_transpose_1_5821conv2d_transpose_1_5823*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_49342,
*conv2d_transpose_1/StatefulPartitionedCall?
0instance_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0instance_normalization_4_5826instance_normalization_4_5828*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_4_layer_call_and_return_conditional_losses_532122
0instance_normalization_4/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall9instance_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_53322
activation_4/PartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_13_5832conv2d_13_5834*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_53442#
!conv2d_13/StatefulPartitionedCall?
0instance_normalization_5/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0instance_normalization_5_5837instance_normalization_5_5839*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_5_layer_call_and_return_conditional_losses_539322
0instance_normalization_5/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall9instance_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_54042
activation_5/PartitionedCall?
IdentityIdentity%activation_5/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall/^instance_normalization/StatefulPartitionedCall1^instance_normalization_1/StatefulPartitionedCall1^instance_normalization_2/StatefulPartitionedCall1^instance_normalization_3/StatefulPartitionedCall1^instance_normalization_4/StatefulPartitionedCall1^instance_normalization_5/StatefulPartitionedCall'^residual_block/StatefulPartitionedCall)^residual_block_1/StatefulPartitionedCall)^residual_block_2/StatefulPartitionedCall)^residual_block_3/StatefulPartitionedCall)^residual_block_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2`
.instance_normalization/StatefulPartitionedCall.instance_normalization/StatefulPartitionedCall2d
0instance_normalization_1/StatefulPartitionedCall0instance_normalization_1/StatefulPartitionedCall2d
0instance_normalization_2/StatefulPartitionedCall0instance_normalization_2/StatefulPartitionedCall2d
0instance_normalization_3/StatefulPartitionedCall0instance_normalization_3/StatefulPartitionedCall2d
0instance_normalization_4/StatefulPartitionedCall0instance_normalization_4/StatefulPartitionedCall2d
0instance_normalization_5/StatefulPartitionedCall0instance_normalization_5/StatefulPartitionedCall2P
&residual_block/StatefulPartitionedCall&residual_block/StatefulPartitionedCall2T
(residual_block_1/StatefulPartitionedCall(residual_block_1/StatefulPartitionedCall2T
(residual_block_2/StatefulPartitionedCall(residual_block_2/StatefulPartitionedCall2T
(residual_block_3/StatefulPartitionedCall(residual_block_3/StatefulPartitionedCall2T
(residual_block_4/StatefulPartitionedCall(residual_block_4/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?/
?
R__inference_instance_normalization_2_layer_call_and_return_conditional_losses_5154

inputs-
reshape_readvariableop_resource:0/
!reshape_1_readvariableop_resource:0
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:?????????02
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:?????????Z?02
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2
moments/variance?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:0*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         0   2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:02	
Reshape?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:0*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         0   2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:02
	Reshape_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????02
batchnorm/addx
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:?????????02
batchnorm/Rsqrt?
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:?????????02
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*0
_output_shapes
:?????????Z?02
batchnorm/mul_1?
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????02
batchnorm/mul_2?
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????02
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*0
_output_shapes
:?????????Z?02
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?

?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5033

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?0
?
R__inference_instance_normalization_4_layer_call_and_return_conditional_losses_5321

inputs-
reshape_readvariableop_resource:/
!reshape_1_readvariableop_resource:
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:?????????2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
moments/variance?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????2
batchnorm/addx
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/Rsqrt?
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:?????????2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2
batchnorm/mul_1?
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/mul_2?
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
J__inference_residual_block_4_layer_call_and_return_conditional_losses_4822
input_1(
conv2d_11_4798:00
conv2d_11_4800:0(
conv2d_12_4814:00
conv2d_12_4816:0
identity??!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_11_4798conv2d_11_4800*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_47972#
!conv2d_11/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_12_4814conv2d_12_4816*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_48132#
!conv2d_12/StatefulPartitionedCall?
add/addAddV2input_1*conv2d_12/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:?????????Z?02	
add/addh

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:?????????Z?02

re_lu/Relu?
IdentityIdentityre_lu/Relu:activations:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????Z?0: : : : 2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall:Y U
0
_output_shapes
:?????????Z?0
!
_user_specified_name	input_1
?
?
'__inference_StyleNet_layer_call_fn_7301

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7: 0
	unknown_8:0
	unknown_9:0

unknown_10:0$

unknown_11:00

unknown_12:0$

unknown_13:00

unknown_14:0$

unknown_15:00

unknown_16:0$

unknown_17:00

unknown_18:0$

unknown_19:00

unknown_20:0$

unknown_21:00

unknown_22:0$

unknown_23:00

unknown_24:0$

unknown_25:00

unknown_26:0$

unknown_27:00

unknown_28:0$

unknown_29:00

unknown_30:0$

unknown_31: 0

unknown_32: 

unknown_33: 

unknown_34: $

unknown_35: 

unknown_36:

unknown_37:

unknown_38:$

unknown_39:

unknown_40:

unknown_41:

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_StyleNet_layer_call_and_return_conditional_losses_58442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
1__inference_conv2d_transpose_1_layer_call_fn_4944

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_49342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
G
+__inference_activation_1_layer_call_fn_7463

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_50932
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?$
?
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4890

inputsB
(conv2d_transpose_readvariableop_resource: 0-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
'__inference_conv2d_7_layer_call_fn_7847

inputs!
unknown:00
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_46492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?

?
B__inference_conv2d_6_layer_call_and_return_conditional_losses_4591

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
??
?
B__inference_StyleNet_layer_call_and_return_conditional_losses_5407

inputs%
conv2d_4962:
conv2d_4964:)
instance_normalization_5011:)
instance_normalization_5013:'
conv2d_1_5034: 
conv2d_1_5036: +
instance_normalization_1_5083: +
instance_normalization_1_5085: '
conv2d_2_5106: 0
conv2d_2_5108:0+
instance_normalization_2_5155:0+
instance_normalization_2_5157:0-
residual_block_5167:00!
residual_block_5169:0-
residual_block_5171:00!
residual_block_5173:0/
residual_block_1_5176:00#
residual_block_1_5178:0/
residual_block_1_5180:00#
residual_block_1_5182:0/
residual_block_2_5185:00#
residual_block_2_5187:0/
residual_block_2_5189:00#
residual_block_2_5191:0/
residual_block_3_5194:00#
residual_block_3_5196:0/
residual_block_3_5198:00#
residual_block_3_5200:0/
residual_block_4_5203:00#
residual_block_4_5205:0/
residual_block_4_5207:00#
residual_block_4_5209:0/
conv2d_transpose_5212: 0#
conv2d_transpose_5214: +
instance_normalization_3_5261: +
instance_normalization_3_5263: 1
conv2d_transpose_1_5273: %
conv2d_transpose_1_5275:+
instance_normalization_4_5322:+
instance_normalization_4_5324:(
conv2d_13_5345:
conv2d_13_5347:+
instance_normalization_5_5394:+
instance_normalization_5_5396:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?.instance_normalization/StatefulPartitionedCall?0instance_normalization_1/StatefulPartitionedCall?0instance_normalization_2/StatefulPartitionedCall?0instance_normalization_3/StatefulPartitionedCall?0instance_normalization_4/StatefulPartitionedCall?0instance_normalization_5/StatefulPartitionedCall?&residual_block/StatefulPartitionedCall?(residual_block_1/StatefulPartitionedCall?(residual_block_2/StatefulPartitionedCall?(residual_block_3/StatefulPartitionedCall?(residual_block_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4962conv2d_4964*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_49612 
conv2d/StatefulPartitionedCall?
.instance_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0instance_normalization_5011instance_normalization_5013*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_instance_normalization_layer_call_and_return_conditional_losses_501020
.instance_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall7instance_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_50212
activation/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_5034conv2d_1_5036*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_50332"
 conv2d_1/StatefulPartitionedCall?
0instance_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0instance_normalization_1_5083instance_normalization_1_5085*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_1_layer_call_and_return_conditional_losses_508222
0instance_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall9instance_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_50932
activation_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_5106conv2d_2_5108*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_51052"
 conv2d_2/StatefulPartitionedCall?
0instance_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0instance_normalization_2_5155instance_normalization_2_5157*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_2_layer_call_and_return_conditional_losses_515422
0instance_normalization_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall9instance_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_51652
activation_2/PartitionedCall?
&residual_block/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0residual_block_5167residual_block_5169residual_block_5171residual_block_5173*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_residual_block_layer_call_and_return_conditional_losses_45262(
&residual_block/StatefulPartitionedCall?
(residual_block_1/StatefulPartitionedCallStatefulPartitionedCall/residual_block/StatefulPartitionedCall:output:0residual_block_1_5176residual_block_1_5178residual_block_1_5180residual_block_1_5182*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_1_layer_call_and_return_conditional_losses_46002*
(residual_block_1/StatefulPartitionedCall?
(residual_block_2/StatefulPartitionedCallStatefulPartitionedCall1residual_block_1/StatefulPartitionedCall:output:0residual_block_2_5185residual_block_2_5187residual_block_2_5189residual_block_2_5191*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_2_layer_call_and_return_conditional_losses_46742*
(residual_block_2/StatefulPartitionedCall?
(residual_block_3/StatefulPartitionedCallStatefulPartitionedCall1residual_block_2/StatefulPartitionedCall:output:0residual_block_3_5194residual_block_3_5196residual_block_3_5198residual_block_3_5200*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_3_layer_call_and_return_conditional_losses_47482*
(residual_block_3/StatefulPartitionedCall?
(residual_block_4/StatefulPartitionedCallStatefulPartitionedCall1residual_block_3/StatefulPartitionedCall:output:0residual_block_4_5203residual_block_4_5205residual_block_4_5207residual_block_4_5209*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_4_layer_call_and_return_conditional_losses_48222*
(residual_block_4/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall1residual_block_4/StatefulPartitionedCall:output:0conv2d_transpose_5212conv2d_transpose_5214*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_48902*
(conv2d_transpose/StatefulPartitionedCall?
0instance_normalization_3/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0instance_normalization_3_5261instance_normalization_3_5263*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_3_layer_call_and_return_conditional_losses_526022
0instance_normalization_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall9instance_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_52712
activation_3/PartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_transpose_1_5273conv2d_transpose_1_5275*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_49342,
*conv2d_transpose_1/StatefulPartitionedCall?
0instance_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0instance_normalization_4_5322instance_normalization_4_5324*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_4_layer_call_and_return_conditional_losses_532122
0instance_normalization_4/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall9instance_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_53322
activation_4/PartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_13_5345conv2d_13_5347*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_53442#
!conv2d_13/StatefulPartitionedCall?
0instance_normalization_5/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0instance_normalization_5_5394instance_normalization_5_5396*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_5_layer_call_and_return_conditional_losses_539322
0instance_normalization_5/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall9instance_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_54042
activation_5/PartitionedCall?
IdentityIdentity%activation_5/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall/^instance_normalization/StatefulPartitionedCall1^instance_normalization_1/StatefulPartitionedCall1^instance_normalization_2/StatefulPartitionedCall1^instance_normalization_3/StatefulPartitionedCall1^instance_normalization_4/StatefulPartitionedCall1^instance_normalization_5/StatefulPartitionedCall'^residual_block/StatefulPartitionedCall)^residual_block_1/StatefulPartitionedCall)^residual_block_2/StatefulPartitionedCall)^residual_block_3/StatefulPartitionedCall)^residual_block_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2`
.instance_normalization/StatefulPartitionedCall.instance_normalization/StatefulPartitionedCall2d
0instance_normalization_1/StatefulPartitionedCall0instance_normalization_1/StatefulPartitionedCall2d
0instance_normalization_2/StatefulPartitionedCall0instance_normalization_2/StatefulPartitionedCall2d
0instance_normalization_3/StatefulPartitionedCall0instance_normalization_3/StatefulPartitionedCall2d
0instance_normalization_4/StatefulPartitionedCall0instance_normalization_4/StatefulPartitionedCall2d
0instance_normalization_5/StatefulPartitionedCall0instance_normalization_5/StatefulPartitionedCall2P
&residual_block/StatefulPartitionedCall&residual_block/StatefulPartitionedCall2T
(residual_block_1/StatefulPartitionedCall(residual_block_1/StatefulPartitionedCall2T
(residual_block_2/StatefulPartitionedCall(residual_block_2/StatefulPartitionedCall2T
(residual_block_3/StatefulPartitionedCall(residual_block_3/StatefulPartitionedCall2T
(residual_block_4/StatefulPartitionedCall(residual_block_4/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?$
?
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4934

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
b
F__inference_activation_1_layer_call_and_return_conditional_losses_7458

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:??????????? 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?

?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_7779

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_4_layer_call_fn_7658

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_4_layer_call_and_return_conditional_losses_53212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_conv2d_6_layer_call_fn_7827

inputs!
unknown:00
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_45912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
B__inference_conv2d_5_layer_call_and_return_conditional_losses_7799

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
'__inference_conv2d_3_layer_call_fn_7769

inputs!
unknown:00
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_45012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
b
F__inference_activation_1_layer_call_and_return_conditional_losses_5093

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:??????????? 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
J__inference_residual_block_3_layer_call_and_return_conditional_losses_4748
input_1'
conv2d_9_4724:00
conv2d_9_4726:0(
conv2d_10_4740:00
conv2d_10_4742:0
identity??!conv2d_10/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_9_4724conv2d_9_4726*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_9_layer_call_and_return_conditional_losses_47232"
 conv2d_9/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_4740conv2d_10_4742*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_47392#
!conv2d_10/StatefulPartitionedCall?
add/addAddV2input_1*conv2d_10/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:?????????Z?02	
add/addh

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:?????????Z?02

re_lu/Relu?
IdentityIdentityre_lu/Relu:activations:0"^conv2d_10/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????Z?0: : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:Y U
0
_output_shapes
:?????????Z?0
!
_user_specified_name	input_1
?
?
7__inference_instance_normalization_5_layer_call_fn_7739

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_5_layer_call_and_return_conditional_losses_53932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_activation_2_layer_call_and_return_conditional_losses_7539

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????Z?02
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Z?0:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
(__inference_conv2d_10_layer_call_fn_7905

inputs!
unknown:00
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_47392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
G
+__inference_activation_2_layer_call_fn_7544

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_51652
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Z?0:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
b
F__inference_activation_5_layer_call_and_return_conditional_losses_7744

inputs
identityh
TanhTanhinputs*
T0*A
_output_shapes/
-:+???????????????????????????2
Tanhv
IdentityIdentityTanh:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_11_layer_call_fn_7925

inputs!
unknown:00
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_47972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?/
?
P__inference_instance_normalization_layer_call_and_return_conditional_losses_5010

inputs-
reshape_readvariableop_resource:/
!reshape_1_readvariableop_resource:
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:?????????2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*1
_output_shapes
:???????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
moments/variance?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????2
batchnorm/addx
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/Rsqrt?
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:?????????2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*1
_output_shapes
:???????????2
batchnorm/mul_1?
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/mul_2?
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*1
_output_shapes
:???????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
'__inference_conv2d_1_layer_call_fn_7401

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_50332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_13_layer_call_and_return_conditional_losses_7678

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?0
?
R__inference_instance_normalization_4_layer_call_and_return_conditional_losses_7649

inputs-
reshape_readvariableop_resource:/
!reshape_1_readvariableop_resource:
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:?????????2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
moments/variance?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????2
batchnorm/addx
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/Rsqrt?
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:?????????2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2
batchnorm/mul_1?
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/mul_2?
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?
 __inference__traced_restore_8241
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:=
/assignvariableop_2_instance_normalization_gamma:<
.assignvariableop_3_instance_normalization_beta:<
"assignvariableop_4_conv2d_1_kernel: .
 assignvariableop_5_conv2d_1_bias: ?
1assignvariableop_6_instance_normalization_1_gamma: >
0assignvariableop_7_instance_normalization_1_beta: <
"assignvariableop_8_conv2d_2_kernel: 0.
 assignvariableop_9_conv2d_2_bias:0@
2assignvariableop_10_instance_normalization_2_gamma:0?
1assignvariableop_11_instance_normalization_2_beta:0E
+assignvariableop_12_conv2d_transpose_kernel: 07
)assignvariableop_13_conv2d_transpose_bias: @
2assignvariableop_14_instance_normalization_3_gamma: ?
1assignvariableop_15_instance_normalization_3_beta: G
-assignvariableop_16_conv2d_transpose_1_kernel: 9
+assignvariableop_17_conv2d_transpose_1_bias:@
2assignvariableop_18_instance_normalization_4_gamma:?
1assignvariableop_19_instance_normalization_4_beta:>
$assignvariableop_20_conv2d_13_kernel:0
"assignvariableop_21_conv2d_13_bias:@
2assignvariableop_22_instance_normalization_5_gamma:?
1assignvariableop_23_instance_normalization_5_beta:L
2assignvariableop_24_residual_block_conv2d_3_kernel:00>
0assignvariableop_25_residual_block_conv2d_3_bias:0L
2assignvariableop_26_residual_block_conv2d_4_kernel:00>
0assignvariableop_27_residual_block_conv2d_4_bias:0N
4assignvariableop_28_residual_block_1_conv2d_5_kernel:00@
2assignvariableop_29_residual_block_1_conv2d_5_bias:0N
4assignvariableop_30_residual_block_1_conv2d_6_kernel:00@
2assignvariableop_31_residual_block_1_conv2d_6_bias:0N
4assignvariableop_32_residual_block_2_conv2d_7_kernel:00@
2assignvariableop_33_residual_block_2_conv2d_7_bias:0N
4assignvariableop_34_residual_block_2_conv2d_8_kernel:00@
2assignvariableop_35_residual_block_2_conv2d_8_bias:0N
4assignvariableop_36_residual_block_3_conv2d_9_kernel:00@
2assignvariableop_37_residual_block_3_conv2d_9_bias:0O
5assignvariableop_38_residual_block_3_conv2d_10_kernel:00A
3assignvariableop_39_residual_block_3_conv2d_10_bias:0O
5assignvariableop_40_residual_block_4_conv2d_11_kernel:00A
3assignvariableop_41_residual_block_4_conv2d_11_bias:0O
5assignvariableop_42_residual_block_4_conv2d_12_kernel:00A
3assignvariableop_43_residual_block_4_conv2d_12_bias:0
identity_45??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*?
value?B?-B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/30/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/31/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_instance_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_instance_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp1assignvariableop_6_instance_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp0assignvariableop_7_instance_normalization_1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp2assignvariableop_10_instance_normalization_2_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp1assignvariableop_11_instance_normalization_2_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp+assignvariableop_12_conv2d_transpose_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp)assignvariableop_13_conv2d_transpose_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp2assignvariableop_14_instance_normalization_3_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp1assignvariableop_15_instance_normalization_3_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp-assignvariableop_16_conv2d_transpose_1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_conv2d_transpose_1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp2assignvariableop_18_instance_normalization_4_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp1assignvariableop_19_instance_normalization_4_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv2d_13_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv2d_13_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp2assignvariableop_22_instance_normalization_5_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp1assignvariableop_23_instance_normalization_5_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp2assignvariableop_24_residual_block_conv2d_3_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp0assignvariableop_25_residual_block_conv2d_3_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp2assignvariableop_26_residual_block_conv2d_4_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp0assignvariableop_27_residual_block_conv2d_4_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp4assignvariableop_28_residual_block_1_conv2d_5_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp2assignvariableop_29_residual_block_1_conv2d_5_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp4assignvariableop_30_residual_block_1_conv2d_6_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp2assignvariableop_31_residual_block_1_conv2d_6_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp4assignvariableop_32_residual_block_2_conv2d_7_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp2assignvariableop_33_residual_block_2_conv2d_7_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp4assignvariableop_34_residual_block_2_conv2d_8_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp2assignvariableop_35_residual_block_2_conv2d_8_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp4assignvariableop_36_residual_block_3_conv2d_9_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp2assignvariableop_37_residual_block_3_conv2d_9_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp5assignvariableop_38_residual_block_3_conv2d_10_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp3assignvariableop_39_residual_block_3_conv2d_10_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp5assignvariableop_40_residual_block_4_conv2d_11_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp3assignvariableop_41_residual_block_4_conv2d_11_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp5assignvariableop_42_residual_block_4_conv2d_12_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp3assignvariableop_43_residual_block_4_conv2d_12_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_439
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_44Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_44?
Identity_45IdentityIdentity_44:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_45"#
identity_45Identity_45:output:0*m
_input_shapes\
Z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
7__inference_instance_normalization_2_layer_call_fn_7534

inputs
unknown:0
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_2_layer_call_and_return_conditional_losses_51542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
J__inference_residual_block_1_layer_call_and_return_conditional_losses_4600
input_1'
conv2d_5_4576:00
conv2d_5_4578:0'
conv2d_6_4592:00
conv2d_6_4594:0
identity?? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_5_4576conv2d_5_4578*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_45752"
 conv2d_5/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_4592conv2d_6_4594*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_45912"
 conv2d_6/StatefulPartitionedCall?
add/addAddV2input_1)conv2d_6/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:?????????Z?02	
add/addh

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:?????????Z?02

re_lu/Relu?
IdentityIdentityre_lu/Relu:activations:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????Z?0: : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:Y U
0
_output_shapes
:?????????Z?0
!
_user_specified_name	input_1
?
b
F__inference_activation_3_layer_call_and_return_conditional_losses_5271

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?
B__inference_StyleNet_layer_call_and_return_conditional_losses_6258
conv2d_input%
conv2d_6146:
conv2d_6148:)
instance_normalization_6151:)
instance_normalization_6153:'
conv2d_1_6157: 
conv2d_1_6159: +
instance_normalization_1_6162: +
instance_normalization_1_6164: '
conv2d_2_6168: 0
conv2d_2_6170:0+
instance_normalization_2_6173:0+
instance_normalization_2_6175:0-
residual_block_6179:00!
residual_block_6181:0-
residual_block_6183:00!
residual_block_6185:0/
residual_block_1_6188:00#
residual_block_1_6190:0/
residual_block_1_6192:00#
residual_block_1_6194:0/
residual_block_2_6197:00#
residual_block_2_6199:0/
residual_block_2_6201:00#
residual_block_2_6203:0/
residual_block_3_6206:00#
residual_block_3_6208:0/
residual_block_3_6210:00#
residual_block_3_6212:0/
residual_block_4_6215:00#
residual_block_4_6217:0/
residual_block_4_6219:00#
residual_block_4_6221:0/
conv2d_transpose_6224: 0#
conv2d_transpose_6226: +
instance_normalization_3_6229: +
instance_normalization_3_6231: 1
conv2d_transpose_1_6235: %
conv2d_transpose_1_6237:+
instance_normalization_4_6240:+
instance_normalization_4_6242:(
conv2d_13_6246:
conv2d_13_6248:+
instance_normalization_5_6251:+
instance_normalization_5_6253:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?.instance_normalization/StatefulPartitionedCall?0instance_normalization_1/StatefulPartitionedCall?0instance_normalization_2/StatefulPartitionedCall?0instance_normalization_3/StatefulPartitionedCall?0instance_normalization_4/StatefulPartitionedCall?0instance_normalization_5/StatefulPartitionedCall?&residual_block/StatefulPartitionedCall?(residual_block_1/StatefulPartitionedCall?(residual_block_2/StatefulPartitionedCall?(residual_block_3/StatefulPartitionedCall?(residual_block_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_6146conv2d_6148*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_49612 
conv2d/StatefulPartitionedCall?
.instance_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0instance_normalization_6151instance_normalization_6153*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_instance_normalization_layer_call_and_return_conditional_losses_501020
.instance_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall7instance_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_50212
activation/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_6157conv2d_1_6159*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_50332"
 conv2d_1/StatefulPartitionedCall?
0instance_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0instance_normalization_1_6162instance_normalization_1_6164*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_1_layer_call_and_return_conditional_losses_508222
0instance_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall9instance_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_50932
activation_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_6168conv2d_2_6170*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_51052"
 conv2d_2/StatefulPartitionedCall?
0instance_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0instance_normalization_2_6173instance_normalization_2_6175*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_2_layer_call_and_return_conditional_losses_515422
0instance_normalization_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall9instance_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_51652
activation_2/PartitionedCall?
&residual_block/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0residual_block_6179residual_block_6181residual_block_6183residual_block_6185*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_residual_block_layer_call_and_return_conditional_losses_45262(
&residual_block/StatefulPartitionedCall?
(residual_block_1/StatefulPartitionedCallStatefulPartitionedCall/residual_block/StatefulPartitionedCall:output:0residual_block_1_6188residual_block_1_6190residual_block_1_6192residual_block_1_6194*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_1_layer_call_and_return_conditional_losses_46002*
(residual_block_1/StatefulPartitionedCall?
(residual_block_2/StatefulPartitionedCallStatefulPartitionedCall1residual_block_1/StatefulPartitionedCall:output:0residual_block_2_6197residual_block_2_6199residual_block_2_6201residual_block_2_6203*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_2_layer_call_and_return_conditional_losses_46742*
(residual_block_2/StatefulPartitionedCall?
(residual_block_3/StatefulPartitionedCallStatefulPartitionedCall1residual_block_2/StatefulPartitionedCall:output:0residual_block_3_6206residual_block_3_6208residual_block_3_6210residual_block_3_6212*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_3_layer_call_and_return_conditional_losses_47482*
(residual_block_3/StatefulPartitionedCall?
(residual_block_4/StatefulPartitionedCallStatefulPartitionedCall1residual_block_3/StatefulPartitionedCall:output:0residual_block_4_6215residual_block_4_6217residual_block_4_6219residual_block_4_6221*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_4_layer_call_and_return_conditional_losses_48222*
(residual_block_4/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall1residual_block_4/StatefulPartitionedCall:output:0conv2d_transpose_6224conv2d_transpose_6226*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_48902*
(conv2d_transpose/StatefulPartitionedCall?
0instance_normalization_3/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0instance_normalization_3_6229instance_normalization_3_6231*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_3_layer_call_and_return_conditional_losses_526022
0instance_normalization_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall9instance_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_52712
activation_3/PartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_transpose_1_6235conv2d_transpose_1_6237*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_49342,
*conv2d_transpose_1/StatefulPartitionedCall?
0instance_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0instance_normalization_4_6240instance_normalization_4_6242*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_4_layer_call_and_return_conditional_losses_532122
0instance_normalization_4/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall9instance_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_53322
activation_4/PartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_13_6246conv2d_13_6248*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_53442#
!conv2d_13/StatefulPartitionedCall?
0instance_normalization_5/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0instance_normalization_5_6251instance_normalization_5_6253*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_5_layer_call_and_return_conditional_losses_539322
0instance_normalization_5/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall9instance_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_54042
activation_5/PartitionedCall?
IdentityIdentity%activation_5/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall/^instance_normalization/StatefulPartitionedCall1^instance_normalization_1/StatefulPartitionedCall1^instance_normalization_2/StatefulPartitionedCall1^instance_normalization_3/StatefulPartitionedCall1^instance_normalization_4/StatefulPartitionedCall1^instance_normalization_5/StatefulPartitionedCall'^residual_block/StatefulPartitionedCall)^residual_block_1/StatefulPartitionedCall)^residual_block_2/StatefulPartitionedCall)^residual_block_3/StatefulPartitionedCall)^residual_block_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2`
.instance_normalization/StatefulPartitionedCall.instance_normalization/StatefulPartitionedCall2d
0instance_normalization_1/StatefulPartitionedCall0instance_normalization_1/StatefulPartitionedCall2d
0instance_normalization_2/StatefulPartitionedCall0instance_normalization_2/StatefulPartitionedCall2d
0instance_normalization_3/StatefulPartitionedCall0instance_normalization_3/StatefulPartitionedCall2d
0instance_normalization_4/StatefulPartitionedCall0instance_normalization_4/StatefulPartitionedCall2d
0instance_normalization_5/StatefulPartitionedCall0instance_normalization_5/StatefulPartitionedCall2P
&residual_block/StatefulPartitionedCall&residual_block/StatefulPartitionedCall2T
(residual_block_1/StatefulPartitionedCall(residual_block_1/StatefulPartitionedCall2T
(residual_block_2/StatefulPartitionedCall(residual_block_2/StatefulPartitionedCall2T
(residual_block_3/StatefulPartitionedCall(residual_block_3/StatefulPartitionedCall2T
(residual_block_4/StatefulPartitionedCall(residual_block_4/StatefulPartitionedCall:_ [
1
_output_shapes
:???????????
&
_user_specified_nameconv2d_input
?

?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_7473

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?

?
@__inference_conv2d_layer_call_and_return_conditional_losses_7311

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_7392

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
%__inference_conv2d_layer_call_fn_7320

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_49612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
'__inference_conv2d_2_layer_call_fn_7482

inputs!
unknown: 0
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_51052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
B__inference_conv2d_9_layer_call_and_return_conditional_losses_7877

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
B__inference_conv2d_7_layer_call_and_return_conditional_losses_7838

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_3_layer_call_fn_7596

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_3_layer_call_and_return_conditional_losses_52602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?0
?
R__inference_instance_normalization_5_layer_call_and_return_conditional_losses_7730

inputs-
reshape_readvariableop_resource:/
!reshape_1_readvariableop_resource:
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:?????????2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
moments/variance?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????2
batchnorm/addx
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/Rsqrt?
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:?????????2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2
batchnorm/mul_1?
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/mul_2?
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?/
?
R__inference_instance_normalization_1_layer_call_and_return_conditional_losses_7444

inputs-
reshape_readvariableop_resource: /
!reshape_1_readvariableop_resource: 
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:????????? 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*1
_output_shapes
:??????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(2
moments/variance?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
: *
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
: 2	
Reshape?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2
	Reshape_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:????????? 2
batchnorm/addx
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:????????? 2
batchnorm/Rsqrt?
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:????????? 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*1
_output_shapes
:??????????? 2
batchnorm/mul_1?
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:????????? 2
batchnorm/mul_2?
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:????????? 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*1
_output_shapes
:??????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
b
F__inference_activation_4_layer_call_and_return_conditional_losses_5332

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?/
?
R__inference_instance_normalization_1_layer_call_and_return_conditional_losses_5082

inputs-
reshape_readvariableop_resource: /
!reshape_1_readvariableop_resource: 
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:????????? 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*1
_output_shapes
:??????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(2
moments/variance?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
: *
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
: 2	
Reshape?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2
	Reshape_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:????????? 2
batchnorm/addx
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:????????? 2
batchnorm/Rsqrt?
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:????????? 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*1
_output_shapes
:??????????? 2
batchnorm/mul_1?
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:????????? 2
batchnorm/mul_2?
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:????????? 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*1
_output_shapes
:??????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
b
F__inference_activation_2_layer_call_and_return_conditional_losses_5165

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????Z?02
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Z?0:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
`
D__inference_activation_layer_call_and_return_conditional_losses_7377

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_4517

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?

?
B__inference_conv2d_6_layer_call_and_return_conditional_losses_7818

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?

?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5105

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
ݖ
?*
B__inference_StyleNet_layer_call_and_return_conditional_losses_6734

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:D
6instance_normalization_reshape_readvariableop_resource:F
8instance_normalization_reshape_1_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: F
8instance_normalization_1_reshape_readvariableop_resource: H
:instance_normalization_1_reshape_1_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: 06
(conv2d_2_biasadd_readvariableop_resource:0F
8instance_normalization_2_reshape_readvariableop_resource:0H
:instance_normalization_2_reshape_1_readvariableop_resource:0P
6residual_block_conv2d_3_conv2d_readvariableop_resource:00E
7residual_block_conv2d_3_biasadd_readvariableop_resource:0P
6residual_block_conv2d_4_conv2d_readvariableop_resource:00E
7residual_block_conv2d_4_biasadd_readvariableop_resource:0R
8residual_block_1_conv2d_5_conv2d_readvariableop_resource:00G
9residual_block_1_conv2d_5_biasadd_readvariableop_resource:0R
8residual_block_1_conv2d_6_conv2d_readvariableop_resource:00G
9residual_block_1_conv2d_6_biasadd_readvariableop_resource:0R
8residual_block_2_conv2d_7_conv2d_readvariableop_resource:00G
9residual_block_2_conv2d_7_biasadd_readvariableop_resource:0R
8residual_block_2_conv2d_8_conv2d_readvariableop_resource:00G
9residual_block_2_conv2d_8_biasadd_readvariableop_resource:0R
8residual_block_3_conv2d_9_conv2d_readvariableop_resource:00G
9residual_block_3_conv2d_9_biasadd_readvariableop_resource:0S
9residual_block_3_conv2d_10_conv2d_readvariableop_resource:00H
:residual_block_3_conv2d_10_biasadd_readvariableop_resource:0S
9residual_block_4_conv2d_11_conv2d_readvariableop_resource:00H
:residual_block_4_conv2d_11_biasadd_readvariableop_resource:0S
9residual_block_4_conv2d_12_conv2d_readvariableop_resource:00H
:residual_block_4_conv2d_12_biasadd_readvariableop_resource:0S
9conv2d_transpose_conv2d_transpose_readvariableop_resource: 0>
0conv2d_transpose_biasadd_readvariableop_resource: F
8instance_normalization_3_reshape_readvariableop_resource: H
:instance_normalization_3_reshape_1_readvariableop_resource: U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_1_biasadd_readvariableop_resource:F
8instance_normalization_4_reshape_readvariableop_resource:H
:instance_normalization_4_reshape_1_readvariableop_resource:B
(conv2d_13_conv2d_readvariableop_resource:7
)conv2d_13_biasadd_readvariableop_resource:F
8instance_normalization_5_reshape_readvariableop_resource:H
:instance_normalization_5_reshape_1_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?-instance_normalization/Reshape/ReadVariableOp?/instance_normalization/Reshape_1/ReadVariableOp?/instance_normalization_1/Reshape/ReadVariableOp?1instance_normalization_1/Reshape_1/ReadVariableOp?/instance_normalization_2/Reshape/ReadVariableOp?1instance_normalization_2/Reshape_1/ReadVariableOp?/instance_normalization_3/Reshape/ReadVariableOp?1instance_normalization_3/Reshape_1/ReadVariableOp?/instance_normalization_4/Reshape/ReadVariableOp?1instance_normalization_4/Reshape_1/ReadVariableOp?/instance_normalization_5/Reshape/ReadVariableOp?1instance_normalization_5/Reshape_1/ReadVariableOp?.residual_block/conv2d_3/BiasAdd/ReadVariableOp?-residual_block/conv2d_3/Conv2D/ReadVariableOp?.residual_block/conv2d_4/BiasAdd/ReadVariableOp?-residual_block/conv2d_4/Conv2D/ReadVariableOp?0residual_block_1/conv2d_5/BiasAdd/ReadVariableOp?/residual_block_1/conv2d_5/Conv2D/ReadVariableOp?0residual_block_1/conv2d_6/BiasAdd/ReadVariableOp?/residual_block_1/conv2d_6/Conv2D/ReadVariableOp?0residual_block_2/conv2d_7/BiasAdd/ReadVariableOp?/residual_block_2/conv2d_7/Conv2D/ReadVariableOp?0residual_block_2/conv2d_8/BiasAdd/ReadVariableOp?/residual_block_2/conv2d_8/Conv2D/ReadVariableOp?1residual_block_3/conv2d_10/BiasAdd/ReadVariableOp?0residual_block_3/conv2d_10/Conv2D/ReadVariableOp?0residual_block_3/conv2d_9/BiasAdd/ReadVariableOp?/residual_block_3/conv2d_9/Conv2D/ReadVariableOp?1residual_block_4/conv2d_11/BiasAdd/ReadVariableOp?0residual_block_4/conv2d_11/Conv2D/ReadVariableOp?1residual_block_4/conv2d_12/BiasAdd/ReadVariableOp?0residual_block_4/conv2d_12/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
instance_normalization/ShapeShapeconv2d/BiasAdd:output:0*
T0*
_output_shapes
:2
instance_normalization/Shape?
*instance_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*instance_normalization/strided_slice/stack?
,instance_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,instance_normalization/strided_slice/stack_1?
,instance_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,instance_normalization/strided_slice/stack_2?
$instance_normalization/strided_sliceStridedSlice%instance_normalization/Shape:output:03instance_normalization/strided_slice/stack:output:05instance_normalization/strided_slice/stack_1:output:05instance_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$instance_normalization/strided_slice?
,instance_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,instance_normalization/strided_slice_1/stack?
.instance_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization/strided_slice_1/stack_1?
.instance_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization/strided_slice_1/stack_2?
&instance_normalization/strided_slice_1StridedSlice%instance_normalization/Shape:output:05instance_normalization/strided_slice_1/stack:output:07instance_normalization/strided_slice_1/stack_1:output:07instance_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&instance_normalization/strided_slice_1?
,instance_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,instance_normalization/strided_slice_2/stack?
.instance_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization/strided_slice_2/stack_1?
.instance_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization/strided_slice_2/stack_2?
&instance_normalization/strided_slice_2StridedSlice%instance_normalization/Shape:output:05instance_normalization/strided_slice_2/stack:output:07instance_normalization/strided_slice_2/stack_1:output:07instance_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&instance_normalization/strided_slice_2?
,instance_normalization/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,instance_normalization/strided_slice_3/stack?
.instance_normalization/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization/strided_slice_3/stack_1?
.instance_normalization/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization/strided_slice_3/stack_2?
&instance_normalization/strided_slice_3StridedSlice%instance_normalization/Shape:output:05instance_normalization/strided_slice_3/stack:output:07instance_normalization/strided_slice_3/stack_1:output:07instance_normalization/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&instance_normalization/strided_slice_3?
5instance_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      27
5instance_normalization/moments/mean/reduction_indices?
#instance_normalization/moments/meanMeanconv2d/BiasAdd:output:0>instance_normalization/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2%
#instance_normalization/moments/mean?
+instance_normalization/moments/StopGradientStopGradient,instance_normalization/moments/mean:output:0*
T0*/
_output_shapes
:?????????2-
+instance_normalization/moments/StopGradient?
0instance_normalization/moments/SquaredDifferenceSquaredDifferenceconv2d/BiasAdd:output:04instance_normalization/moments/StopGradient:output:0*
T0*1
_output_shapes
:???????????22
0instance_normalization/moments/SquaredDifference?
9instance_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2;
9instance_normalization/moments/variance/reduction_indices?
'instance_normalization/moments/varianceMean4instance_normalization/moments/SquaredDifference:z:0Binstance_normalization/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2)
'instance_normalization/moments/variance?
-instance_normalization/Reshape/ReadVariableOpReadVariableOp6instance_normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02/
-instance_normalization/Reshape/ReadVariableOp?
$instance_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2&
$instance_normalization/Reshape/shape?
instance_normalization/ReshapeReshape5instance_normalization/Reshape/ReadVariableOp:value:0-instance_normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2 
instance_normalization/Reshape?
/instance_normalization/Reshape_1/ReadVariableOpReadVariableOp8instance_normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype021
/instance_normalization/Reshape_1/ReadVariableOp?
&instance_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2(
&instance_normalization/Reshape_1/shape?
 instance_normalization/Reshape_1Reshape7instance_normalization/Reshape_1/ReadVariableOp:value:0/instance_normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2"
 instance_normalization/Reshape_1?
&instance_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&instance_normalization/batchnorm/add/y?
$instance_normalization/batchnorm/addAddV20instance_normalization/moments/variance:output:0/instance_normalization/batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????2&
$instance_normalization/batchnorm/add?
&instance_normalization/batchnorm/RsqrtRsqrt(instance_normalization/batchnorm/add:z:0*
T0*/
_output_shapes
:?????????2(
&instance_normalization/batchnorm/Rsqrt?
$instance_normalization/batchnorm/mulMul*instance_normalization/batchnorm/Rsqrt:y:0'instance_normalization/Reshape:output:0*
T0*/
_output_shapes
:?????????2&
$instance_normalization/batchnorm/mul?
&instance_normalization/batchnorm/mul_1Mulconv2d/BiasAdd:output:0(instance_normalization/batchnorm/mul:z:0*
T0*1
_output_shapes
:???????????2(
&instance_normalization/batchnorm/mul_1?
&instance_normalization/batchnorm/mul_2Mul,instance_normalization/moments/mean:output:0(instance_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????2(
&instance_normalization/batchnorm/mul_2?
$instance_normalization/batchnorm/subSub)instance_normalization/Reshape_1:output:0*instance_normalization/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????2&
$instance_normalization/batchnorm/sub?
&instance_normalization/batchnorm/add_1AddV2*instance_normalization/batchnorm/mul_1:z:0(instance_normalization/batchnorm/sub:z:0*
T0*1
_output_shapes
:???????????2(
&instance_normalization/batchnorm/add_1?
activation/ReluRelu*instance_normalization/batchnorm/add_1:z:0*
T0*1
_output_shapes
:???????????2
activation/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dactivation/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_1/BiasAdd?
instance_normalization_1/ShapeShapeconv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2 
instance_normalization_1/Shape?
,instance_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,instance_normalization_1/strided_slice/stack?
.instance_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_1/strided_slice/stack_1?
.instance_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_1/strided_slice/stack_2?
&instance_normalization_1/strided_sliceStridedSlice'instance_normalization_1/Shape:output:05instance_normalization_1/strided_slice/stack:output:07instance_normalization_1/strided_slice/stack_1:output:07instance_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&instance_normalization_1/strided_slice?
.instance_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_1/strided_slice_1/stack?
0instance_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_1/strided_slice_1/stack_1?
0instance_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_1/strided_slice_1/stack_2?
(instance_normalization_1/strided_slice_1StridedSlice'instance_normalization_1/Shape:output:07instance_normalization_1/strided_slice_1/stack:output:09instance_normalization_1/strided_slice_1/stack_1:output:09instance_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_1/strided_slice_1?
.instance_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_1/strided_slice_2/stack?
0instance_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_1/strided_slice_2/stack_1?
0instance_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_1/strided_slice_2/stack_2?
(instance_normalization_1/strided_slice_2StridedSlice'instance_normalization_1/Shape:output:07instance_normalization_1/strided_slice_2/stack:output:09instance_normalization_1/strided_slice_2/stack_1:output:09instance_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_1/strided_slice_2?
.instance_normalization_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_1/strided_slice_3/stack?
0instance_normalization_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_1/strided_slice_3/stack_1?
0instance_normalization_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_1/strided_slice_3/stack_2?
(instance_normalization_1/strided_slice_3StridedSlice'instance_normalization_1/Shape:output:07instance_normalization_1/strided_slice_3/stack:output:09instance_normalization_1/strided_slice_3/stack_1:output:09instance_normalization_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_1/strided_slice_3?
7instance_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7instance_normalization_1/moments/mean/reduction_indices?
%instance_normalization_1/moments/meanMeanconv2d_1/BiasAdd:output:0@instance_normalization_1/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(2'
%instance_normalization_1/moments/mean?
-instance_normalization_1/moments/StopGradientStopGradient.instance_normalization_1/moments/mean:output:0*
T0*/
_output_shapes
:????????? 2/
-instance_normalization_1/moments/StopGradient?
2instance_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv2d_1/BiasAdd:output:06instance_normalization_1/moments/StopGradient:output:0*
T0*1
_output_shapes
:??????????? 24
2instance_normalization_1/moments/SquaredDifference?
;instance_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2=
;instance_normalization_1/moments/variance/reduction_indices?
)instance_normalization_1/moments/varianceMean6instance_normalization_1/moments/SquaredDifference:z:0Dinstance_normalization_1/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(2+
)instance_normalization_1/moments/variance?
/instance_normalization_1/Reshape/ReadVariableOpReadVariableOp8instance_normalization_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype021
/instance_normalization_1/Reshape/ReadVariableOp?
&instance_normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&instance_normalization_1/Reshape/shape?
 instance_normalization_1/ReshapeReshape7instance_normalization_1/Reshape/ReadVariableOp:value:0/instance_normalization_1/Reshape/shape:output:0*
T0*&
_output_shapes
: 2"
 instance_normalization_1/Reshape?
1instance_normalization_1/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_1_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype023
1instance_normalization_1/Reshape_1/ReadVariableOp?
(instance_normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2*
(instance_normalization_1/Reshape_1/shape?
"instance_normalization_1/Reshape_1Reshape9instance_normalization_1/Reshape_1/ReadVariableOp:value:01instance_normalization_1/Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2$
"instance_normalization_1/Reshape_1?
(instance_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2*
(instance_normalization_1/batchnorm/add/y?
&instance_normalization_1/batchnorm/addAddV22instance_normalization_1/moments/variance:output:01instance_normalization_1/batchnorm/add/y:output:0*
T0*/
_output_shapes
:????????? 2(
&instance_normalization_1/batchnorm/add?
(instance_normalization_1/batchnorm/RsqrtRsqrt*instance_normalization_1/batchnorm/add:z:0*
T0*/
_output_shapes
:????????? 2*
(instance_normalization_1/batchnorm/Rsqrt?
&instance_normalization_1/batchnorm/mulMul,instance_normalization_1/batchnorm/Rsqrt:y:0)instance_normalization_1/Reshape:output:0*
T0*/
_output_shapes
:????????? 2(
&instance_normalization_1/batchnorm/mul?
(instance_normalization_1/batchnorm/mul_1Mulconv2d_1/BiasAdd:output:0*instance_normalization_1/batchnorm/mul:z:0*
T0*1
_output_shapes
:??????????? 2*
(instance_normalization_1/batchnorm/mul_1?
(instance_normalization_1/batchnorm/mul_2Mul.instance_normalization_1/moments/mean:output:0*instance_normalization_1/batchnorm/mul:z:0*
T0*/
_output_shapes
:????????? 2*
(instance_normalization_1/batchnorm/mul_2?
&instance_normalization_1/batchnorm/subSub+instance_normalization_1/Reshape_1:output:0,instance_normalization_1/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:????????? 2(
&instance_normalization_1/batchnorm/sub?
(instance_normalization_1/batchnorm/add_1AddV2,instance_normalization_1/batchnorm/mul_1:z:0*instance_normalization_1/batchnorm/sub:z:0*
T0*1
_output_shapes
:??????????? 2*
(instance_normalization_1/batchnorm/add_1?
activation_1/ReluRelu,instance_normalization_1/batchnorm/add_1:z:0*
T0*1
_output_shapes
:??????????? 2
activation_1/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02
conv2d_2/BiasAdd?
instance_normalization_2/ShapeShapeconv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:2 
instance_normalization_2/Shape?
,instance_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,instance_normalization_2/strided_slice/stack?
.instance_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_2/strided_slice/stack_1?
.instance_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_2/strided_slice/stack_2?
&instance_normalization_2/strided_sliceStridedSlice'instance_normalization_2/Shape:output:05instance_normalization_2/strided_slice/stack:output:07instance_normalization_2/strided_slice/stack_1:output:07instance_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&instance_normalization_2/strided_slice?
.instance_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_2/strided_slice_1/stack?
0instance_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_2/strided_slice_1/stack_1?
0instance_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_2/strided_slice_1/stack_2?
(instance_normalization_2/strided_slice_1StridedSlice'instance_normalization_2/Shape:output:07instance_normalization_2/strided_slice_1/stack:output:09instance_normalization_2/strided_slice_1/stack_1:output:09instance_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_2/strided_slice_1?
.instance_normalization_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_2/strided_slice_2/stack?
0instance_normalization_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_2/strided_slice_2/stack_1?
0instance_normalization_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_2/strided_slice_2/stack_2?
(instance_normalization_2/strided_slice_2StridedSlice'instance_normalization_2/Shape:output:07instance_normalization_2/strided_slice_2/stack:output:09instance_normalization_2/strided_slice_2/stack_1:output:09instance_normalization_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_2/strided_slice_2?
.instance_normalization_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_2/strided_slice_3/stack?
0instance_normalization_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_2/strided_slice_3/stack_1?
0instance_normalization_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_2/strided_slice_3/stack_2?
(instance_normalization_2/strided_slice_3StridedSlice'instance_normalization_2/Shape:output:07instance_normalization_2/strided_slice_3/stack:output:09instance_normalization_2/strided_slice_3/stack_1:output:09instance_normalization_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_2/strided_slice_3?
7instance_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7instance_normalization_2/moments/mean/reduction_indices?
%instance_normalization_2/moments/meanMeanconv2d_2/BiasAdd:output:0@instance_normalization_2/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2'
%instance_normalization_2/moments/mean?
-instance_normalization_2/moments/StopGradientStopGradient.instance_normalization_2/moments/mean:output:0*
T0*/
_output_shapes
:?????????02/
-instance_normalization_2/moments/StopGradient?
2instance_normalization_2/moments/SquaredDifferenceSquaredDifferenceconv2d_2/BiasAdd:output:06instance_normalization_2/moments/StopGradient:output:0*
T0*0
_output_shapes
:?????????Z?024
2instance_normalization_2/moments/SquaredDifference?
;instance_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2=
;instance_normalization_2/moments/variance/reduction_indices?
)instance_normalization_2/moments/varianceMean6instance_normalization_2/moments/SquaredDifference:z:0Dinstance_normalization_2/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2+
)instance_normalization_2/moments/variance?
/instance_normalization_2/Reshape/ReadVariableOpReadVariableOp8instance_normalization_2_reshape_readvariableop_resource*
_output_shapes
:0*
dtype021
/instance_normalization_2/Reshape/ReadVariableOp?
&instance_normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         0   2(
&instance_normalization_2/Reshape/shape?
 instance_normalization_2/ReshapeReshape7instance_normalization_2/Reshape/ReadVariableOp:value:0/instance_normalization_2/Reshape/shape:output:0*
T0*&
_output_shapes
:02"
 instance_normalization_2/Reshape?
1instance_normalization_2/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_2_reshape_1_readvariableop_resource*
_output_shapes
:0*
dtype023
1instance_normalization_2/Reshape_1/ReadVariableOp?
(instance_normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         0   2*
(instance_normalization_2/Reshape_1/shape?
"instance_normalization_2/Reshape_1Reshape9instance_normalization_2/Reshape_1/ReadVariableOp:value:01instance_normalization_2/Reshape_1/shape:output:0*
T0*&
_output_shapes
:02$
"instance_normalization_2/Reshape_1?
(instance_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2*
(instance_normalization_2/batchnorm/add/y?
&instance_normalization_2/batchnorm/addAddV22instance_normalization_2/moments/variance:output:01instance_normalization_2/batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????02(
&instance_normalization_2/batchnorm/add?
(instance_normalization_2/batchnorm/RsqrtRsqrt*instance_normalization_2/batchnorm/add:z:0*
T0*/
_output_shapes
:?????????02*
(instance_normalization_2/batchnorm/Rsqrt?
&instance_normalization_2/batchnorm/mulMul,instance_normalization_2/batchnorm/Rsqrt:y:0)instance_normalization_2/Reshape:output:0*
T0*/
_output_shapes
:?????????02(
&instance_normalization_2/batchnorm/mul?
(instance_normalization_2/batchnorm/mul_1Mulconv2d_2/BiasAdd:output:0*instance_normalization_2/batchnorm/mul:z:0*
T0*0
_output_shapes
:?????????Z?02*
(instance_normalization_2/batchnorm/mul_1?
(instance_normalization_2/batchnorm/mul_2Mul.instance_normalization_2/moments/mean:output:0*instance_normalization_2/batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????02*
(instance_normalization_2/batchnorm/mul_2?
&instance_normalization_2/batchnorm/subSub+instance_normalization_2/Reshape_1:output:0,instance_normalization_2/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????02(
&instance_normalization_2/batchnorm/sub?
(instance_normalization_2/batchnorm/add_1AddV2,instance_normalization_2/batchnorm/mul_1:z:0*instance_normalization_2/batchnorm/sub:z:0*
T0*0
_output_shapes
:?????????Z?02*
(instance_normalization_2/batchnorm/add_1?
activation_2/ReluRelu,instance_normalization_2/batchnorm/add_1:z:0*
T0*0
_output_shapes
:?????????Z?02
activation_2/Relu?
-residual_block/conv2d_3/Conv2D/ReadVariableOpReadVariableOp6residual_block_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02/
-residual_block/conv2d_3/Conv2D/ReadVariableOp?
residual_block/conv2d_3/Conv2DConv2Dactivation_2/Relu:activations:05residual_block/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2 
residual_block/conv2d_3/Conv2D?
.residual_block/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp7residual_block_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype020
.residual_block/conv2d_3/BiasAdd/ReadVariableOp?
residual_block/conv2d_3/BiasAddBiasAdd'residual_block/conv2d_3/Conv2D:output:06residual_block/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02!
residual_block/conv2d_3/BiasAdd?
residual_block/conv2d_3/ReluRelu(residual_block/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
residual_block/conv2d_3/Relu?
-residual_block/conv2d_4/Conv2D/ReadVariableOpReadVariableOp6residual_block_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02/
-residual_block/conv2d_4/Conv2D/ReadVariableOp?
residual_block/conv2d_4/Conv2DConv2D*residual_block/conv2d_3/Relu:activations:05residual_block/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2 
residual_block/conv2d_4/Conv2D?
.residual_block/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp7residual_block_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype020
.residual_block/conv2d_4/BiasAdd/ReadVariableOp?
residual_block/conv2d_4/BiasAddBiasAdd'residual_block/conv2d_4/Conv2D:output:06residual_block/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02!
residual_block/conv2d_4/BiasAdd?
residual_block/add/addAddV2activation_2/Relu:activations:0(residual_block/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
residual_block/add/add?
residual_block/re_lu/ReluReluresidual_block/add/add:z:0*
T0*0
_output_shapes
:?????????Z?02
residual_block/re_lu/Relu?
/residual_block_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp8residual_block_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype021
/residual_block_1/conv2d_5/Conv2D/ReadVariableOp?
 residual_block_1/conv2d_5/Conv2DConv2D'residual_block/re_lu/Relu:activations:07residual_block_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2"
 residual_block_1/conv2d_5/Conv2D?
0residual_block_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype022
0residual_block_1/conv2d_5/BiasAdd/ReadVariableOp?
!residual_block_1/conv2d_5/BiasAddBiasAdd)residual_block_1/conv2d_5/Conv2D:output:08residual_block_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02#
!residual_block_1/conv2d_5/BiasAdd?
residual_block_1/conv2d_5/ReluRelu*residual_block_1/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02 
residual_block_1/conv2d_5/Relu?
/residual_block_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp8residual_block_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype021
/residual_block_1/conv2d_6/Conv2D/ReadVariableOp?
 residual_block_1/conv2d_6/Conv2DConv2D,residual_block_1/conv2d_5/Relu:activations:07residual_block_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2"
 residual_block_1/conv2d_6/Conv2D?
0residual_block_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype022
0residual_block_1/conv2d_6/BiasAdd/ReadVariableOp?
!residual_block_1/conv2d_6/BiasAddBiasAdd)residual_block_1/conv2d_6/Conv2D:output:08residual_block_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02#
!residual_block_1/conv2d_6/BiasAdd?
residual_block_1/add_1/addAddV2'residual_block/re_lu/Relu:activations:0*residual_block_1/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
residual_block_1/add_1/add?
residual_block_1/re_lu_1/ReluReluresidual_block_1/add_1/add:z:0*
T0*0
_output_shapes
:?????????Z?02
residual_block_1/re_lu_1/Relu?
/residual_block_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp8residual_block_2_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype021
/residual_block_2/conv2d_7/Conv2D/ReadVariableOp?
 residual_block_2/conv2d_7/Conv2DConv2D+residual_block_1/re_lu_1/Relu:activations:07residual_block_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2"
 residual_block_2/conv2d_7/Conv2D?
0residual_block_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype022
0residual_block_2/conv2d_7/BiasAdd/ReadVariableOp?
!residual_block_2/conv2d_7/BiasAddBiasAdd)residual_block_2/conv2d_7/Conv2D:output:08residual_block_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02#
!residual_block_2/conv2d_7/BiasAdd?
residual_block_2/conv2d_7/ReluRelu*residual_block_2/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02 
residual_block_2/conv2d_7/Relu?
/residual_block_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp8residual_block_2_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype021
/residual_block_2/conv2d_8/Conv2D/ReadVariableOp?
 residual_block_2/conv2d_8/Conv2DConv2D,residual_block_2/conv2d_7/Relu:activations:07residual_block_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2"
 residual_block_2/conv2d_8/Conv2D?
0residual_block_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype022
0residual_block_2/conv2d_8/BiasAdd/ReadVariableOp?
!residual_block_2/conv2d_8/BiasAddBiasAdd)residual_block_2/conv2d_8/Conv2D:output:08residual_block_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02#
!residual_block_2/conv2d_8/BiasAdd?
residual_block_2/add_2/addAddV2+residual_block_1/re_lu_1/Relu:activations:0*residual_block_2/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
residual_block_2/add_2/add?
residual_block_2/re_lu_2/ReluReluresidual_block_2/add_2/add:z:0*
T0*0
_output_shapes
:?????????Z?02
residual_block_2/re_lu_2/Relu?
/residual_block_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp8residual_block_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype021
/residual_block_3/conv2d_9/Conv2D/ReadVariableOp?
 residual_block_3/conv2d_9/Conv2DConv2D+residual_block_2/re_lu_2/Relu:activations:07residual_block_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2"
 residual_block_3/conv2d_9/Conv2D?
0residual_block_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype022
0residual_block_3/conv2d_9/BiasAdd/ReadVariableOp?
!residual_block_3/conv2d_9/BiasAddBiasAdd)residual_block_3/conv2d_9/Conv2D:output:08residual_block_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02#
!residual_block_3/conv2d_9/BiasAdd?
residual_block_3/conv2d_9/ReluRelu*residual_block_3/conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02 
residual_block_3/conv2d_9/Relu?
0residual_block_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp9residual_block_3_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype022
0residual_block_3/conv2d_10/Conv2D/ReadVariableOp?
!residual_block_3/conv2d_10/Conv2DConv2D,residual_block_3/conv2d_9/Relu:activations:08residual_block_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2#
!residual_block_3/conv2d_10/Conv2D?
1residual_block_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp:residual_block_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype023
1residual_block_3/conv2d_10/BiasAdd/ReadVariableOp?
"residual_block_3/conv2d_10/BiasAddBiasAdd*residual_block_3/conv2d_10/Conv2D:output:09residual_block_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02$
"residual_block_3/conv2d_10/BiasAdd?
residual_block_3/add_3/addAddV2+residual_block_2/re_lu_2/Relu:activations:0+residual_block_3/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
residual_block_3/add_3/add?
residual_block_3/re_lu_3/ReluReluresidual_block_3/add_3/add:z:0*
T0*0
_output_shapes
:?????????Z?02
residual_block_3/re_lu_3/Relu?
0residual_block_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOp9residual_block_4_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype022
0residual_block_4/conv2d_11/Conv2D/ReadVariableOp?
!residual_block_4/conv2d_11/Conv2DConv2D+residual_block_3/re_lu_3/Relu:activations:08residual_block_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2#
!residual_block_4/conv2d_11/Conv2D?
1residual_block_4/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp:residual_block_4_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype023
1residual_block_4/conv2d_11/BiasAdd/ReadVariableOp?
"residual_block_4/conv2d_11/BiasAddBiasAdd*residual_block_4/conv2d_11/Conv2D:output:09residual_block_4/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02$
"residual_block_4/conv2d_11/BiasAdd?
residual_block_4/conv2d_11/ReluRelu+residual_block_4/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02!
residual_block_4/conv2d_11/Relu?
0residual_block_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp9residual_block_4_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype022
0residual_block_4/conv2d_12/Conv2D/ReadVariableOp?
!residual_block_4/conv2d_12/Conv2DConv2D-residual_block_4/conv2d_11/Relu:activations:08residual_block_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2#
!residual_block_4/conv2d_12/Conv2D?
1residual_block_4/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp:residual_block_4_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype023
1residual_block_4/conv2d_12/BiasAdd/ReadVariableOp?
"residual_block_4/conv2d_12/BiasAddBiasAdd*residual_block_4/conv2d_12/Conv2D:output:09residual_block_4/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02$
"residual_block_4/conv2d_12/BiasAdd?
residual_block_4/add_4/addAddV2+residual_block_3/re_lu_3/Relu:activations:0+residual_block_4/conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
residual_block_4/add_4/add?
residual_block_4/re_lu_4/ReluReluresidual_block_4/add_4/add:z:0*
T0*0
_output_shapes
:?????????Z?02
residual_block_4/re_lu_4/Relu?
conv2d_transpose/ShapeShape+residual_block_4/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicew
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/1w
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0+residual_block_4/re_lu_4/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_transpose/BiasAdd?
instance_normalization_3/ShapeShape!conv2d_transpose/BiasAdd:output:0*
T0*
_output_shapes
:2 
instance_normalization_3/Shape?
,instance_normalization_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,instance_normalization_3/strided_slice/stack?
.instance_normalization_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_3/strided_slice/stack_1?
.instance_normalization_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_3/strided_slice/stack_2?
&instance_normalization_3/strided_sliceStridedSlice'instance_normalization_3/Shape:output:05instance_normalization_3/strided_slice/stack:output:07instance_normalization_3/strided_slice/stack_1:output:07instance_normalization_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&instance_normalization_3/strided_slice?
.instance_normalization_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_3/strided_slice_1/stack?
0instance_normalization_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_3/strided_slice_1/stack_1?
0instance_normalization_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_3/strided_slice_1/stack_2?
(instance_normalization_3/strided_slice_1StridedSlice'instance_normalization_3/Shape:output:07instance_normalization_3/strided_slice_1/stack:output:09instance_normalization_3/strided_slice_1/stack_1:output:09instance_normalization_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_3/strided_slice_1?
.instance_normalization_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_3/strided_slice_2/stack?
0instance_normalization_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_3/strided_slice_2/stack_1?
0instance_normalization_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_3/strided_slice_2/stack_2?
(instance_normalization_3/strided_slice_2StridedSlice'instance_normalization_3/Shape:output:07instance_normalization_3/strided_slice_2/stack:output:09instance_normalization_3/strided_slice_2/stack_1:output:09instance_normalization_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_3/strided_slice_2?
.instance_normalization_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_3/strided_slice_3/stack?
0instance_normalization_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_3/strided_slice_3/stack_1?
0instance_normalization_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_3/strided_slice_3/stack_2?
(instance_normalization_3/strided_slice_3StridedSlice'instance_normalization_3/Shape:output:07instance_normalization_3/strided_slice_3/stack:output:09instance_normalization_3/strided_slice_3/stack_1:output:09instance_normalization_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_3/strided_slice_3?
7instance_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7instance_normalization_3/moments/mean/reduction_indices?
%instance_normalization_3/moments/meanMean!conv2d_transpose/BiasAdd:output:0@instance_normalization_3/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(2'
%instance_normalization_3/moments/mean?
-instance_normalization_3/moments/StopGradientStopGradient.instance_normalization_3/moments/mean:output:0*
T0*/
_output_shapes
:????????? 2/
-instance_normalization_3/moments/StopGradient?
2instance_normalization_3/moments/SquaredDifferenceSquaredDifference!conv2d_transpose/BiasAdd:output:06instance_normalization_3/moments/StopGradient:output:0*
T0*1
_output_shapes
:??????????? 24
2instance_normalization_3/moments/SquaredDifference?
;instance_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2=
;instance_normalization_3/moments/variance/reduction_indices?
)instance_normalization_3/moments/varianceMean6instance_normalization_3/moments/SquaredDifference:z:0Dinstance_normalization_3/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(2+
)instance_normalization_3/moments/variance?
/instance_normalization_3/Reshape/ReadVariableOpReadVariableOp8instance_normalization_3_reshape_readvariableop_resource*
_output_shapes
: *
dtype021
/instance_normalization_3/Reshape/ReadVariableOp?
&instance_normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&instance_normalization_3/Reshape/shape?
 instance_normalization_3/ReshapeReshape7instance_normalization_3/Reshape/ReadVariableOp:value:0/instance_normalization_3/Reshape/shape:output:0*
T0*&
_output_shapes
: 2"
 instance_normalization_3/Reshape?
1instance_normalization_3/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_3_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype023
1instance_normalization_3/Reshape_1/ReadVariableOp?
(instance_normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2*
(instance_normalization_3/Reshape_1/shape?
"instance_normalization_3/Reshape_1Reshape9instance_normalization_3/Reshape_1/ReadVariableOp:value:01instance_normalization_3/Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2$
"instance_normalization_3/Reshape_1?
(instance_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2*
(instance_normalization_3/batchnorm/add/y?
&instance_normalization_3/batchnorm/addAddV22instance_normalization_3/moments/variance:output:01instance_normalization_3/batchnorm/add/y:output:0*
T0*/
_output_shapes
:????????? 2(
&instance_normalization_3/batchnorm/add?
(instance_normalization_3/batchnorm/RsqrtRsqrt*instance_normalization_3/batchnorm/add:z:0*
T0*/
_output_shapes
:????????? 2*
(instance_normalization_3/batchnorm/Rsqrt?
&instance_normalization_3/batchnorm/mulMul,instance_normalization_3/batchnorm/Rsqrt:y:0)instance_normalization_3/Reshape:output:0*
T0*/
_output_shapes
:????????? 2(
&instance_normalization_3/batchnorm/mul?
(instance_normalization_3/batchnorm/mul_1Mul!conv2d_transpose/BiasAdd:output:0*instance_normalization_3/batchnorm/mul:z:0*
T0*1
_output_shapes
:??????????? 2*
(instance_normalization_3/batchnorm/mul_1?
(instance_normalization_3/batchnorm/mul_2Mul.instance_normalization_3/moments/mean:output:0*instance_normalization_3/batchnorm/mul:z:0*
T0*/
_output_shapes
:????????? 2*
(instance_normalization_3/batchnorm/mul_2?
&instance_normalization_3/batchnorm/subSub+instance_normalization_3/Reshape_1:output:0,instance_normalization_3/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:????????? 2(
&instance_normalization_3/batchnorm/sub?
(instance_normalization_3/batchnorm/add_1AddV2,instance_normalization_3/batchnorm/mul_1:z:0*instance_normalization_3/batchnorm/sub:z:0*
T0*1
_output_shapes
:??????????? 2*
(instance_normalization_3/batchnorm/add_1?
activation_3/ReluRelu,instance_normalization_3/batchnorm/add_1:z:0*
T0*1
_output_shapes
:??????????? 2
activation_3/Relu?
conv2d_transpose_1/ShapeShapeactivation_3/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice{
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_1/stack/1{
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0activation_3/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_1/BiasAdd?
instance_normalization_4/ShapeShape#conv2d_transpose_1/BiasAdd:output:0*
T0*
_output_shapes
:2 
instance_normalization_4/Shape?
,instance_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,instance_normalization_4/strided_slice/stack?
.instance_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_4/strided_slice/stack_1?
.instance_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_4/strided_slice/stack_2?
&instance_normalization_4/strided_sliceStridedSlice'instance_normalization_4/Shape:output:05instance_normalization_4/strided_slice/stack:output:07instance_normalization_4/strided_slice/stack_1:output:07instance_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&instance_normalization_4/strided_slice?
.instance_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_4/strided_slice_1/stack?
0instance_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_4/strided_slice_1/stack_1?
0instance_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_4/strided_slice_1/stack_2?
(instance_normalization_4/strided_slice_1StridedSlice'instance_normalization_4/Shape:output:07instance_normalization_4/strided_slice_1/stack:output:09instance_normalization_4/strided_slice_1/stack_1:output:09instance_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_4/strided_slice_1?
.instance_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_4/strided_slice_2/stack?
0instance_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_4/strided_slice_2/stack_1?
0instance_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_4/strided_slice_2/stack_2?
(instance_normalization_4/strided_slice_2StridedSlice'instance_normalization_4/Shape:output:07instance_normalization_4/strided_slice_2/stack:output:09instance_normalization_4/strided_slice_2/stack_1:output:09instance_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_4/strided_slice_2?
.instance_normalization_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_4/strided_slice_3/stack?
0instance_normalization_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_4/strided_slice_3/stack_1?
0instance_normalization_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_4/strided_slice_3/stack_2?
(instance_normalization_4/strided_slice_3StridedSlice'instance_normalization_4/Shape:output:07instance_normalization_4/strided_slice_3/stack:output:09instance_normalization_4/strided_slice_3/stack_1:output:09instance_normalization_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_4/strided_slice_3?
7instance_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7instance_normalization_4/moments/mean/reduction_indices?
%instance_normalization_4/moments/meanMean#conv2d_transpose_1/BiasAdd:output:0@instance_normalization_4/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2'
%instance_normalization_4/moments/mean?
-instance_normalization_4/moments/StopGradientStopGradient.instance_normalization_4/moments/mean:output:0*
T0*/
_output_shapes
:?????????2/
-instance_normalization_4/moments/StopGradient?
2instance_normalization_4/moments/SquaredDifferenceSquaredDifference#conv2d_transpose_1/BiasAdd:output:06instance_normalization_4/moments/StopGradient:output:0*
T0*1
_output_shapes
:???????????24
2instance_normalization_4/moments/SquaredDifference?
;instance_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2=
;instance_normalization_4/moments/variance/reduction_indices?
)instance_normalization_4/moments/varianceMean6instance_normalization_4/moments/SquaredDifference:z:0Dinstance_normalization_4/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2+
)instance_normalization_4/moments/variance?
/instance_normalization_4/Reshape/ReadVariableOpReadVariableOp8instance_normalization_4_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/instance_normalization_4/Reshape/ReadVariableOp?
&instance_normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2(
&instance_normalization_4/Reshape/shape?
 instance_normalization_4/ReshapeReshape7instance_normalization_4/Reshape/ReadVariableOp:value:0/instance_normalization_4/Reshape/shape:output:0*
T0*&
_output_shapes
:2"
 instance_normalization_4/Reshape?
1instance_normalization_4/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_4_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1instance_normalization_4/Reshape_1/ReadVariableOp?
(instance_normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2*
(instance_normalization_4/Reshape_1/shape?
"instance_normalization_4/Reshape_1Reshape9instance_normalization_4/Reshape_1/ReadVariableOp:value:01instance_normalization_4/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2$
"instance_normalization_4/Reshape_1?
(instance_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2*
(instance_normalization_4/batchnorm/add/y?
&instance_normalization_4/batchnorm/addAddV22instance_normalization_4/moments/variance:output:01instance_normalization_4/batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????2(
&instance_normalization_4/batchnorm/add?
(instance_normalization_4/batchnorm/RsqrtRsqrt*instance_normalization_4/batchnorm/add:z:0*
T0*/
_output_shapes
:?????????2*
(instance_normalization_4/batchnorm/Rsqrt?
&instance_normalization_4/batchnorm/mulMul,instance_normalization_4/batchnorm/Rsqrt:y:0)instance_normalization_4/Reshape:output:0*
T0*/
_output_shapes
:?????????2(
&instance_normalization_4/batchnorm/mul?
(instance_normalization_4/batchnorm/mul_1Mul#conv2d_transpose_1/BiasAdd:output:0*instance_normalization_4/batchnorm/mul:z:0*
T0*1
_output_shapes
:???????????2*
(instance_normalization_4/batchnorm/mul_1?
(instance_normalization_4/batchnorm/mul_2Mul.instance_normalization_4/moments/mean:output:0*instance_normalization_4/batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????2*
(instance_normalization_4/batchnorm/mul_2?
&instance_normalization_4/batchnorm/subSub+instance_normalization_4/Reshape_1:output:0,instance_normalization_4/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????2(
&instance_normalization_4/batchnorm/sub?
(instance_normalization_4/batchnorm/add_1AddV2,instance_normalization_4/batchnorm/mul_1:z:0*instance_normalization_4/batchnorm/sub:z:0*
T0*1
_output_shapes
:???????????2*
(instance_normalization_4/batchnorm/add_1?
activation_4/ReluRelu,instance_normalization_4/batchnorm/add_1:z:0*
T0*1
_output_shapes
:???????????2
activation_4/Relu?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2Dactivation_4/Relu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_13/Conv2D?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_13/BiasAdd?
instance_normalization_5/ShapeShapeconv2d_13/BiasAdd:output:0*
T0*
_output_shapes
:2 
instance_normalization_5/Shape?
,instance_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,instance_normalization_5/strided_slice/stack?
.instance_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_5/strided_slice/stack_1?
.instance_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_5/strided_slice/stack_2?
&instance_normalization_5/strided_sliceStridedSlice'instance_normalization_5/Shape:output:05instance_normalization_5/strided_slice/stack:output:07instance_normalization_5/strided_slice/stack_1:output:07instance_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&instance_normalization_5/strided_slice?
.instance_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_5/strided_slice_1/stack?
0instance_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_5/strided_slice_1/stack_1?
0instance_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_5/strided_slice_1/stack_2?
(instance_normalization_5/strided_slice_1StridedSlice'instance_normalization_5/Shape:output:07instance_normalization_5/strided_slice_1/stack:output:09instance_normalization_5/strided_slice_1/stack_1:output:09instance_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_5/strided_slice_1?
.instance_normalization_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_5/strided_slice_2/stack?
0instance_normalization_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_5/strided_slice_2/stack_1?
0instance_normalization_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_5/strided_slice_2/stack_2?
(instance_normalization_5/strided_slice_2StridedSlice'instance_normalization_5/Shape:output:07instance_normalization_5/strided_slice_2/stack:output:09instance_normalization_5/strided_slice_2/stack_1:output:09instance_normalization_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_5/strided_slice_2?
.instance_normalization_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_5/strided_slice_3/stack?
0instance_normalization_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_5/strided_slice_3/stack_1?
0instance_normalization_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_5/strided_slice_3/stack_2?
(instance_normalization_5/strided_slice_3StridedSlice'instance_normalization_5/Shape:output:07instance_normalization_5/strided_slice_3/stack:output:09instance_normalization_5/strided_slice_3/stack_1:output:09instance_normalization_5/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_5/strided_slice_3?
7instance_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7instance_normalization_5/moments/mean/reduction_indices?
%instance_normalization_5/moments/meanMeanconv2d_13/BiasAdd:output:0@instance_normalization_5/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2'
%instance_normalization_5/moments/mean?
-instance_normalization_5/moments/StopGradientStopGradient.instance_normalization_5/moments/mean:output:0*
T0*/
_output_shapes
:?????????2/
-instance_normalization_5/moments/StopGradient?
2instance_normalization_5/moments/SquaredDifferenceSquaredDifferenceconv2d_13/BiasAdd:output:06instance_normalization_5/moments/StopGradient:output:0*
T0*1
_output_shapes
:???????????24
2instance_normalization_5/moments/SquaredDifference?
;instance_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2=
;instance_normalization_5/moments/variance/reduction_indices?
)instance_normalization_5/moments/varianceMean6instance_normalization_5/moments/SquaredDifference:z:0Dinstance_normalization_5/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2+
)instance_normalization_5/moments/variance?
/instance_normalization_5/Reshape/ReadVariableOpReadVariableOp8instance_normalization_5_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/instance_normalization_5/Reshape/ReadVariableOp?
&instance_normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2(
&instance_normalization_5/Reshape/shape?
 instance_normalization_5/ReshapeReshape7instance_normalization_5/Reshape/ReadVariableOp:value:0/instance_normalization_5/Reshape/shape:output:0*
T0*&
_output_shapes
:2"
 instance_normalization_5/Reshape?
1instance_normalization_5/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_5_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1instance_normalization_5/Reshape_1/ReadVariableOp?
(instance_normalization_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2*
(instance_normalization_5/Reshape_1/shape?
"instance_normalization_5/Reshape_1Reshape9instance_normalization_5/Reshape_1/ReadVariableOp:value:01instance_normalization_5/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2$
"instance_normalization_5/Reshape_1?
(instance_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2*
(instance_normalization_5/batchnorm/add/y?
&instance_normalization_5/batchnorm/addAddV22instance_normalization_5/moments/variance:output:01instance_normalization_5/batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????2(
&instance_normalization_5/batchnorm/add?
(instance_normalization_5/batchnorm/RsqrtRsqrt*instance_normalization_5/batchnorm/add:z:0*
T0*/
_output_shapes
:?????????2*
(instance_normalization_5/batchnorm/Rsqrt?
&instance_normalization_5/batchnorm/mulMul,instance_normalization_5/batchnorm/Rsqrt:y:0)instance_normalization_5/Reshape:output:0*
T0*/
_output_shapes
:?????????2(
&instance_normalization_5/batchnorm/mul?
(instance_normalization_5/batchnorm/mul_1Mulconv2d_13/BiasAdd:output:0*instance_normalization_5/batchnorm/mul:z:0*
T0*1
_output_shapes
:???????????2*
(instance_normalization_5/batchnorm/mul_1?
(instance_normalization_5/batchnorm/mul_2Mul.instance_normalization_5/moments/mean:output:0*instance_normalization_5/batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????2*
(instance_normalization_5/batchnorm/mul_2?
&instance_normalization_5/batchnorm/subSub+instance_normalization_5/Reshape_1:output:0,instance_normalization_5/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????2(
&instance_normalization_5/batchnorm/sub?
(instance_normalization_5/batchnorm/add_1AddV2,instance_normalization_5/batchnorm/mul_1:z:0*instance_normalization_5/batchnorm/sub:z:0*
T0*1
_output_shapes
:???????????2*
(instance_normalization_5/batchnorm/add_1?
activation_5/TanhTanh,instance_normalization_5/batchnorm/add_1:z:0*
T0*1
_output_shapes
:???????????2
activation_5/Tanh?
IdentityIdentityactivation_5/Tanh:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp.^instance_normalization/Reshape/ReadVariableOp0^instance_normalization/Reshape_1/ReadVariableOp0^instance_normalization_1/Reshape/ReadVariableOp2^instance_normalization_1/Reshape_1/ReadVariableOp0^instance_normalization_2/Reshape/ReadVariableOp2^instance_normalization_2/Reshape_1/ReadVariableOp0^instance_normalization_3/Reshape/ReadVariableOp2^instance_normalization_3/Reshape_1/ReadVariableOp0^instance_normalization_4/Reshape/ReadVariableOp2^instance_normalization_4/Reshape_1/ReadVariableOp0^instance_normalization_5/Reshape/ReadVariableOp2^instance_normalization_5/Reshape_1/ReadVariableOp/^residual_block/conv2d_3/BiasAdd/ReadVariableOp.^residual_block/conv2d_3/Conv2D/ReadVariableOp/^residual_block/conv2d_4/BiasAdd/ReadVariableOp.^residual_block/conv2d_4/Conv2D/ReadVariableOp1^residual_block_1/conv2d_5/BiasAdd/ReadVariableOp0^residual_block_1/conv2d_5/Conv2D/ReadVariableOp1^residual_block_1/conv2d_6/BiasAdd/ReadVariableOp0^residual_block_1/conv2d_6/Conv2D/ReadVariableOp1^residual_block_2/conv2d_7/BiasAdd/ReadVariableOp0^residual_block_2/conv2d_7/Conv2D/ReadVariableOp1^residual_block_2/conv2d_8/BiasAdd/ReadVariableOp0^residual_block_2/conv2d_8/Conv2D/ReadVariableOp2^residual_block_3/conv2d_10/BiasAdd/ReadVariableOp1^residual_block_3/conv2d_10/Conv2D/ReadVariableOp1^residual_block_3/conv2d_9/BiasAdd/ReadVariableOp0^residual_block_3/conv2d_9/Conv2D/ReadVariableOp2^residual_block_4/conv2d_11/BiasAdd/ReadVariableOp1^residual_block_4/conv2d_11/Conv2D/ReadVariableOp2^residual_block_4/conv2d_12/BiasAdd/ReadVariableOp1^residual_block_4/conv2d_12/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^
-instance_normalization/Reshape/ReadVariableOp-instance_normalization/Reshape/ReadVariableOp2b
/instance_normalization/Reshape_1/ReadVariableOp/instance_normalization/Reshape_1/ReadVariableOp2b
/instance_normalization_1/Reshape/ReadVariableOp/instance_normalization_1/Reshape/ReadVariableOp2f
1instance_normalization_1/Reshape_1/ReadVariableOp1instance_normalization_1/Reshape_1/ReadVariableOp2b
/instance_normalization_2/Reshape/ReadVariableOp/instance_normalization_2/Reshape/ReadVariableOp2f
1instance_normalization_2/Reshape_1/ReadVariableOp1instance_normalization_2/Reshape_1/ReadVariableOp2b
/instance_normalization_3/Reshape/ReadVariableOp/instance_normalization_3/Reshape/ReadVariableOp2f
1instance_normalization_3/Reshape_1/ReadVariableOp1instance_normalization_3/Reshape_1/ReadVariableOp2b
/instance_normalization_4/Reshape/ReadVariableOp/instance_normalization_4/Reshape/ReadVariableOp2f
1instance_normalization_4/Reshape_1/ReadVariableOp1instance_normalization_4/Reshape_1/ReadVariableOp2b
/instance_normalization_5/Reshape/ReadVariableOp/instance_normalization_5/Reshape/ReadVariableOp2f
1instance_normalization_5/Reshape_1/ReadVariableOp1instance_normalization_5/Reshape_1/ReadVariableOp2`
.residual_block/conv2d_3/BiasAdd/ReadVariableOp.residual_block/conv2d_3/BiasAdd/ReadVariableOp2^
-residual_block/conv2d_3/Conv2D/ReadVariableOp-residual_block/conv2d_3/Conv2D/ReadVariableOp2`
.residual_block/conv2d_4/BiasAdd/ReadVariableOp.residual_block/conv2d_4/BiasAdd/ReadVariableOp2^
-residual_block/conv2d_4/Conv2D/ReadVariableOp-residual_block/conv2d_4/Conv2D/ReadVariableOp2d
0residual_block_1/conv2d_5/BiasAdd/ReadVariableOp0residual_block_1/conv2d_5/BiasAdd/ReadVariableOp2b
/residual_block_1/conv2d_5/Conv2D/ReadVariableOp/residual_block_1/conv2d_5/Conv2D/ReadVariableOp2d
0residual_block_1/conv2d_6/BiasAdd/ReadVariableOp0residual_block_1/conv2d_6/BiasAdd/ReadVariableOp2b
/residual_block_1/conv2d_6/Conv2D/ReadVariableOp/residual_block_1/conv2d_6/Conv2D/ReadVariableOp2d
0residual_block_2/conv2d_7/BiasAdd/ReadVariableOp0residual_block_2/conv2d_7/BiasAdd/ReadVariableOp2b
/residual_block_2/conv2d_7/Conv2D/ReadVariableOp/residual_block_2/conv2d_7/Conv2D/ReadVariableOp2d
0residual_block_2/conv2d_8/BiasAdd/ReadVariableOp0residual_block_2/conv2d_8/BiasAdd/ReadVariableOp2b
/residual_block_2/conv2d_8/Conv2D/ReadVariableOp/residual_block_2/conv2d_8/Conv2D/ReadVariableOp2f
1residual_block_3/conv2d_10/BiasAdd/ReadVariableOp1residual_block_3/conv2d_10/BiasAdd/ReadVariableOp2d
0residual_block_3/conv2d_10/Conv2D/ReadVariableOp0residual_block_3/conv2d_10/Conv2D/ReadVariableOp2d
0residual_block_3/conv2d_9/BiasAdd/ReadVariableOp0residual_block_3/conv2d_9/BiasAdd/ReadVariableOp2b
/residual_block_3/conv2d_9/Conv2D/ReadVariableOp/residual_block_3/conv2d_9/Conv2D/ReadVariableOp2f
1residual_block_4/conv2d_11/BiasAdd/ReadVariableOp1residual_block_4/conv2d_11/BiasAdd/ReadVariableOp2d
0residual_block_4/conv2d_11/Conv2D/ReadVariableOp0residual_block_4/conv2d_11/Conv2D/ReadVariableOp2f
1residual_block_4/conv2d_12/BiasAdd/ReadVariableOp1residual_block_4/conv2d_12/BiasAdd/ReadVariableOp2d
0residual_block_4/conv2d_12/Conv2D/ReadVariableOp0residual_block_4/conv2d_12/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
G
+__inference_activation_4_layer_call_fn_7668

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_53322
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
/__inference_residual_block_1_layer_call_fn_4614
input_1!
unknown:00
	unknown_0:0#
	unknown_1:00
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_1_layer_call_and_return_conditional_losses_46002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????Z?0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????Z?0
!
_user_specified_name	input_1
?
?
'__inference_StyleNet_layer_call_fn_5498
conv2d_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7: 0
	unknown_8:0
	unknown_9:0

unknown_10:0$

unknown_11:00

unknown_12:0$

unknown_13:00

unknown_14:0$

unknown_15:00

unknown_16:0$

unknown_17:00

unknown_18:0$

unknown_19:00

unknown_20:0$

unknown_21:00

unknown_22:0$

unknown_23:00

unknown_24:0$

unknown_25:00

unknown_26:0$

unknown_27:00

unknown_28:0$

unknown_29:00

unknown_30:0$

unknown_31: 0

unknown_32: 

unknown_33: 

unknown_34: $

unknown_35: 

unknown_36:

unknown_37:

unknown_38:$

unknown_39:

unknown_40:

unknown_41:

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_StyleNet_layer_call_and_return_conditional_losses_54072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:???????????
&
_user_specified_nameconv2d_input
?

?
C__inference_conv2d_10_layer_call_and_return_conditional_losses_4739

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
??
?
B__inference_StyleNet_layer_call_and_return_conditional_losses_6143
conv2d_input%
conv2d_6031:
conv2d_6033:)
instance_normalization_6036:)
instance_normalization_6038:'
conv2d_1_6042: 
conv2d_1_6044: +
instance_normalization_1_6047: +
instance_normalization_1_6049: '
conv2d_2_6053: 0
conv2d_2_6055:0+
instance_normalization_2_6058:0+
instance_normalization_2_6060:0-
residual_block_6064:00!
residual_block_6066:0-
residual_block_6068:00!
residual_block_6070:0/
residual_block_1_6073:00#
residual_block_1_6075:0/
residual_block_1_6077:00#
residual_block_1_6079:0/
residual_block_2_6082:00#
residual_block_2_6084:0/
residual_block_2_6086:00#
residual_block_2_6088:0/
residual_block_3_6091:00#
residual_block_3_6093:0/
residual_block_3_6095:00#
residual_block_3_6097:0/
residual_block_4_6100:00#
residual_block_4_6102:0/
residual_block_4_6104:00#
residual_block_4_6106:0/
conv2d_transpose_6109: 0#
conv2d_transpose_6111: +
instance_normalization_3_6114: +
instance_normalization_3_6116: 1
conv2d_transpose_1_6120: %
conv2d_transpose_1_6122:+
instance_normalization_4_6125:+
instance_normalization_4_6127:(
conv2d_13_6131:
conv2d_13_6133:+
instance_normalization_5_6136:+
instance_normalization_5_6138:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?.instance_normalization/StatefulPartitionedCall?0instance_normalization_1/StatefulPartitionedCall?0instance_normalization_2/StatefulPartitionedCall?0instance_normalization_3/StatefulPartitionedCall?0instance_normalization_4/StatefulPartitionedCall?0instance_normalization_5/StatefulPartitionedCall?&residual_block/StatefulPartitionedCall?(residual_block_1/StatefulPartitionedCall?(residual_block_2/StatefulPartitionedCall?(residual_block_3/StatefulPartitionedCall?(residual_block_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_6031conv2d_6033*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_49612 
conv2d/StatefulPartitionedCall?
.instance_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0instance_normalization_6036instance_normalization_6038*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_instance_normalization_layer_call_and_return_conditional_losses_501020
.instance_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall7instance_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_50212
activation/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_6042conv2d_1_6044*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_50332"
 conv2d_1/StatefulPartitionedCall?
0instance_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0instance_normalization_1_6047instance_normalization_1_6049*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_1_layer_call_and_return_conditional_losses_508222
0instance_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall9instance_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_50932
activation_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_6053conv2d_2_6055*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_51052"
 conv2d_2/StatefulPartitionedCall?
0instance_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0instance_normalization_2_6058instance_normalization_2_6060*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_2_layer_call_and_return_conditional_losses_515422
0instance_normalization_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall9instance_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_51652
activation_2/PartitionedCall?
&residual_block/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0residual_block_6064residual_block_6066residual_block_6068residual_block_6070*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_residual_block_layer_call_and_return_conditional_losses_45262(
&residual_block/StatefulPartitionedCall?
(residual_block_1/StatefulPartitionedCallStatefulPartitionedCall/residual_block/StatefulPartitionedCall:output:0residual_block_1_6073residual_block_1_6075residual_block_1_6077residual_block_1_6079*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_1_layer_call_and_return_conditional_losses_46002*
(residual_block_1/StatefulPartitionedCall?
(residual_block_2/StatefulPartitionedCallStatefulPartitionedCall1residual_block_1/StatefulPartitionedCall:output:0residual_block_2_6082residual_block_2_6084residual_block_2_6086residual_block_2_6088*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_2_layer_call_and_return_conditional_losses_46742*
(residual_block_2/StatefulPartitionedCall?
(residual_block_3/StatefulPartitionedCallStatefulPartitionedCall1residual_block_2/StatefulPartitionedCall:output:0residual_block_3_6091residual_block_3_6093residual_block_3_6095residual_block_3_6097*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_3_layer_call_and_return_conditional_losses_47482*
(residual_block_3/StatefulPartitionedCall?
(residual_block_4/StatefulPartitionedCallStatefulPartitionedCall1residual_block_3/StatefulPartitionedCall:output:0residual_block_4_6100residual_block_4_6102residual_block_4_6104residual_block_4_6106*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_4_layer_call_and_return_conditional_losses_48222*
(residual_block_4/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall1residual_block_4/StatefulPartitionedCall:output:0conv2d_transpose_6109conv2d_transpose_6111*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_48902*
(conv2d_transpose/StatefulPartitionedCall?
0instance_normalization_3/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0instance_normalization_3_6114instance_normalization_3_6116*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_3_layer_call_and_return_conditional_losses_526022
0instance_normalization_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall9instance_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_52712
activation_3/PartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_transpose_1_6120conv2d_transpose_1_6122*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_49342,
*conv2d_transpose_1/StatefulPartitionedCall?
0instance_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0instance_normalization_4_6125instance_normalization_4_6127*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_4_layer_call_and_return_conditional_losses_532122
0instance_normalization_4/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall9instance_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_53322
activation_4/PartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_13_6131conv2d_13_6133*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_53442#
!conv2d_13/StatefulPartitionedCall?
0instance_normalization_5/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0instance_normalization_5_6136instance_normalization_5_6138*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_5_layer_call_and_return_conditional_losses_539322
0instance_normalization_5/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall9instance_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_54042
activation_5/PartitionedCall?
IdentityIdentity%activation_5/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall/^instance_normalization/StatefulPartitionedCall1^instance_normalization_1/StatefulPartitionedCall1^instance_normalization_2/StatefulPartitionedCall1^instance_normalization_3/StatefulPartitionedCall1^instance_normalization_4/StatefulPartitionedCall1^instance_normalization_5/StatefulPartitionedCall'^residual_block/StatefulPartitionedCall)^residual_block_1/StatefulPartitionedCall)^residual_block_2/StatefulPartitionedCall)^residual_block_3/StatefulPartitionedCall)^residual_block_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2`
.instance_normalization/StatefulPartitionedCall.instance_normalization/StatefulPartitionedCall2d
0instance_normalization_1/StatefulPartitionedCall0instance_normalization_1/StatefulPartitionedCall2d
0instance_normalization_2/StatefulPartitionedCall0instance_normalization_2/StatefulPartitionedCall2d
0instance_normalization_3/StatefulPartitionedCall0instance_normalization_3/StatefulPartitionedCall2d
0instance_normalization_4/StatefulPartitionedCall0instance_normalization_4/StatefulPartitionedCall2d
0instance_normalization_5/StatefulPartitionedCall0instance_normalization_5/StatefulPartitionedCall2P
&residual_block/StatefulPartitionedCall&residual_block/StatefulPartitionedCall2T
(residual_block_1/StatefulPartitionedCall(residual_block_1/StatefulPartitionedCall2T
(residual_block_2/StatefulPartitionedCall(residual_block_2/StatefulPartitionedCall2T
(residual_block_3/StatefulPartitionedCall(residual_block_3/StatefulPartitionedCall2T
(residual_block_4/StatefulPartitionedCall(residual_block_4/StatefulPartitionedCall:_ [
1
_output_shapes
:???????????
&
_user_specified_nameconv2d_input
??
?0
__inference__wrapped_model_4486
conv2d_inputH
.stylenet_conv2d_conv2d_readvariableop_resource:=
/stylenet_conv2d_biasadd_readvariableop_resource:M
?stylenet_instance_normalization_reshape_readvariableop_resource:O
Astylenet_instance_normalization_reshape_1_readvariableop_resource:J
0stylenet_conv2d_1_conv2d_readvariableop_resource: ?
1stylenet_conv2d_1_biasadd_readvariableop_resource: O
Astylenet_instance_normalization_1_reshape_readvariableop_resource: Q
Cstylenet_instance_normalization_1_reshape_1_readvariableop_resource: J
0stylenet_conv2d_2_conv2d_readvariableop_resource: 0?
1stylenet_conv2d_2_biasadd_readvariableop_resource:0O
Astylenet_instance_normalization_2_reshape_readvariableop_resource:0Q
Cstylenet_instance_normalization_2_reshape_1_readvariableop_resource:0Y
?stylenet_residual_block_conv2d_3_conv2d_readvariableop_resource:00N
@stylenet_residual_block_conv2d_3_biasadd_readvariableop_resource:0Y
?stylenet_residual_block_conv2d_4_conv2d_readvariableop_resource:00N
@stylenet_residual_block_conv2d_4_biasadd_readvariableop_resource:0[
Astylenet_residual_block_1_conv2d_5_conv2d_readvariableop_resource:00P
Bstylenet_residual_block_1_conv2d_5_biasadd_readvariableop_resource:0[
Astylenet_residual_block_1_conv2d_6_conv2d_readvariableop_resource:00P
Bstylenet_residual_block_1_conv2d_6_biasadd_readvariableop_resource:0[
Astylenet_residual_block_2_conv2d_7_conv2d_readvariableop_resource:00P
Bstylenet_residual_block_2_conv2d_7_biasadd_readvariableop_resource:0[
Astylenet_residual_block_2_conv2d_8_conv2d_readvariableop_resource:00P
Bstylenet_residual_block_2_conv2d_8_biasadd_readvariableop_resource:0[
Astylenet_residual_block_3_conv2d_9_conv2d_readvariableop_resource:00P
Bstylenet_residual_block_3_conv2d_9_biasadd_readvariableop_resource:0\
Bstylenet_residual_block_3_conv2d_10_conv2d_readvariableop_resource:00Q
Cstylenet_residual_block_3_conv2d_10_biasadd_readvariableop_resource:0\
Bstylenet_residual_block_4_conv2d_11_conv2d_readvariableop_resource:00Q
Cstylenet_residual_block_4_conv2d_11_biasadd_readvariableop_resource:0\
Bstylenet_residual_block_4_conv2d_12_conv2d_readvariableop_resource:00Q
Cstylenet_residual_block_4_conv2d_12_biasadd_readvariableop_resource:0\
Bstylenet_conv2d_transpose_conv2d_transpose_readvariableop_resource: 0G
9stylenet_conv2d_transpose_biasadd_readvariableop_resource: O
Astylenet_instance_normalization_3_reshape_readvariableop_resource: Q
Cstylenet_instance_normalization_3_reshape_1_readvariableop_resource: ^
Dstylenet_conv2d_transpose_1_conv2d_transpose_readvariableop_resource: I
;stylenet_conv2d_transpose_1_biasadd_readvariableop_resource:O
Astylenet_instance_normalization_4_reshape_readvariableop_resource:Q
Cstylenet_instance_normalization_4_reshape_1_readvariableop_resource:K
1stylenet_conv2d_13_conv2d_readvariableop_resource:@
2stylenet_conv2d_13_biasadd_readvariableop_resource:O
Astylenet_instance_normalization_5_reshape_readvariableop_resource:Q
Cstylenet_instance_normalization_5_reshape_1_readvariableop_resource:
identity??&StyleNet/conv2d/BiasAdd/ReadVariableOp?%StyleNet/conv2d/Conv2D/ReadVariableOp?(StyleNet/conv2d_1/BiasAdd/ReadVariableOp?'StyleNet/conv2d_1/Conv2D/ReadVariableOp?)StyleNet/conv2d_13/BiasAdd/ReadVariableOp?(StyleNet/conv2d_13/Conv2D/ReadVariableOp?(StyleNet/conv2d_2/BiasAdd/ReadVariableOp?'StyleNet/conv2d_2/Conv2D/ReadVariableOp?0StyleNet/conv2d_transpose/BiasAdd/ReadVariableOp?9StyleNet/conv2d_transpose/conv2d_transpose/ReadVariableOp?2StyleNet/conv2d_transpose_1/BiasAdd/ReadVariableOp?;StyleNet/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6StyleNet/instance_normalization/Reshape/ReadVariableOp?8StyleNet/instance_normalization/Reshape_1/ReadVariableOp?8StyleNet/instance_normalization_1/Reshape/ReadVariableOp?:StyleNet/instance_normalization_1/Reshape_1/ReadVariableOp?8StyleNet/instance_normalization_2/Reshape/ReadVariableOp?:StyleNet/instance_normalization_2/Reshape_1/ReadVariableOp?8StyleNet/instance_normalization_3/Reshape/ReadVariableOp?:StyleNet/instance_normalization_3/Reshape_1/ReadVariableOp?8StyleNet/instance_normalization_4/Reshape/ReadVariableOp?:StyleNet/instance_normalization_4/Reshape_1/ReadVariableOp?8StyleNet/instance_normalization_5/Reshape/ReadVariableOp?:StyleNet/instance_normalization_5/Reshape_1/ReadVariableOp?7StyleNet/residual_block/conv2d_3/BiasAdd/ReadVariableOp?6StyleNet/residual_block/conv2d_3/Conv2D/ReadVariableOp?7StyleNet/residual_block/conv2d_4/BiasAdd/ReadVariableOp?6StyleNet/residual_block/conv2d_4/Conv2D/ReadVariableOp?9StyleNet/residual_block_1/conv2d_5/BiasAdd/ReadVariableOp?8StyleNet/residual_block_1/conv2d_5/Conv2D/ReadVariableOp?9StyleNet/residual_block_1/conv2d_6/BiasAdd/ReadVariableOp?8StyleNet/residual_block_1/conv2d_6/Conv2D/ReadVariableOp?9StyleNet/residual_block_2/conv2d_7/BiasAdd/ReadVariableOp?8StyleNet/residual_block_2/conv2d_7/Conv2D/ReadVariableOp?9StyleNet/residual_block_2/conv2d_8/BiasAdd/ReadVariableOp?8StyleNet/residual_block_2/conv2d_8/Conv2D/ReadVariableOp?:StyleNet/residual_block_3/conv2d_10/BiasAdd/ReadVariableOp?9StyleNet/residual_block_3/conv2d_10/Conv2D/ReadVariableOp?9StyleNet/residual_block_3/conv2d_9/BiasAdd/ReadVariableOp?8StyleNet/residual_block_3/conv2d_9/Conv2D/ReadVariableOp?:StyleNet/residual_block_4/conv2d_11/BiasAdd/ReadVariableOp?9StyleNet/residual_block_4/conv2d_11/Conv2D/ReadVariableOp?:StyleNet/residual_block_4/conv2d_12/BiasAdd/ReadVariableOp?9StyleNet/residual_block_4/conv2d_12/Conv2D/ReadVariableOp?
%StyleNet/conv2d/Conv2D/ReadVariableOpReadVariableOp.stylenet_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%StyleNet/conv2d/Conv2D/ReadVariableOp?
StyleNet/conv2d/Conv2DConv2Dconv2d_input-StyleNet/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
StyleNet/conv2d/Conv2D?
&StyleNet/conv2d/BiasAdd/ReadVariableOpReadVariableOp/stylenet_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&StyleNet/conv2d/BiasAdd/ReadVariableOp?
StyleNet/conv2d/BiasAddBiasAddStyleNet/conv2d/Conv2D:output:0.StyleNet/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
StyleNet/conv2d/BiasAdd?
%StyleNet/instance_normalization/ShapeShape StyleNet/conv2d/BiasAdd:output:0*
T0*
_output_shapes
:2'
%StyleNet/instance_normalization/Shape?
3StyleNet/instance_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3StyleNet/instance_normalization/strided_slice/stack?
5StyleNet/instance_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5StyleNet/instance_normalization/strided_slice/stack_1?
5StyleNet/instance_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5StyleNet/instance_normalization/strided_slice/stack_2?
-StyleNet/instance_normalization/strided_sliceStridedSlice.StyleNet/instance_normalization/Shape:output:0<StyleNet/instance_normalization/strided_slice/stack:output:0>StyleNet/instance_normalization/strided_slice/stack_1:output:0>StyleNet/instance_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-StyleNet/instance_normalization/strided_slice?
5StyleNet/instance_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5StyleNet/instance_normalization/strided_slice_1/stack?
7StyleNet/instance_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization/strided_slice_1/stack_1?
7StyleNet/instance_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization/strided_slice_1/stack_2?
/StyleNet/instance_normalization/strided_slice_1StridedSlice.StyleNet/instance_normalization/Shape:output:0>StyleNet/instance_normalization/strided_slice_1/stack:output:0@StyleNet/instance_normalization/strided_slice_1/stack_1:output:0@StyleNet/instance_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/StyleNet/instance_normalization/strided_slice_1?
5StyleNet/instance_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5StyleNet/instance_normalization/strided_slice_2/stack?
7StyleNet/instance_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization/strided_slice_2/stack_1?
7StyleNet/instance_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization/strided_slice_2/stack_2?
/StyleNet/instance_normalization/strided_slice_2StridedSlice.StyleNet/instance_normalization/Shape:output:0>StyleNet/instance_normalization/strided_slice_2/stack:output:0@StyleNet/instance_normalization/strided_slice_2/stack_1:output:0@StyleNet/instance_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/StyleNet/instance_normalization/strided_slice_2?
5StyleNet/instance_normalization/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5StyleNet/instance_normalization/strided_slice_3/stack?
7StyleNet/instance_normalization/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization/strided_slice_3/stack_1?
7StyleNet/instance_normalization/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization/strided_slice_3/stack_2?
/StyleNet/instance_normalization/strided_slice_3StridedSlice.StyleNet/instance_normalization/Shape:output:0>StyleNet/instance_normalization/strided_slice_3/stack:output:0@StyleNet/instance_normalization/strided_slice_3/stack_1:output:0@StyleNet/instance_normalization/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/StyleNet/instance_normalization/strided_slice_3?
>StyleNet/instance_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2@
>StyleNet/instance_normalization/moments/mean/reduction_indices?
,StyleNet/instance_normalization/moments/meanMean StyleNet/conv2d/BiasAdd:output:0GStyleNet/instance_normalization/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2.
,StyleNet/instance_normalization/moments/mean?
4StyleNet/instance_normalization/moments/StopGradientStopGradient5StyleNet/instance_normalization/moments/mean:output:0*
T0*/
_output_shapes
:?????????26
4StyleNet/instance_normalization/moments/StopGradient?
9StyleNet/instance_normalization/moments/SquaredDifferenceSquaredDifference StyleNet/conv2d/BiasAdd:output:0=StyleNet/instance_normalization/moments/StopGradient:output:0*
T0*1
_output_shapes
:???????????2;
9StyleNet/instance_normalization/moments/SquaredDifference?
BStyleNet/instance_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2D
BStyleNet/instance_normalization/moments/variance/reduction_indices?
0StyleNet/instance_normalization/moments/varianceMean=StyleNet/instance_normalization/moments/SquaredDifference:z:0KStyleNet/instance_normalization/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(22
0StyleNet/instance_normalization/moments/variance?
6StyleNet/instance_normalization/Reshape/ReadVariableOpReadVariableOp?stylenet_instance_normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype028
6StyleNet/instance_normalization/Reshape/ReadVariableOp?
-StyleNet/instance_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2/
-StyleNet/instance_normalization/Reshape/shape?
'StyleNet/instance_normalization/ReshapeReshape>StyleNet/instance_normalization/Reshape/ReadVariableOp:value:06StyleNet/instance_normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2)
'StyleNet/instance_normalization/Reshape?
8StyleNet/instance_normalization/Reshape_1/ReadVariableOpReadVariableOpAstylenet_instance_normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02:
8StyleNet/instance_normalization/Reshape_1/ReadVariableOp?
/StyleNet/instance_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/StyleNet/instance_normalization/Reshape_1/shape?
)StyleNet/instance_normalization/Reshape_1Reshape@StyleNet/instance_normalization/Reshape_1/ReadVariableOp:value:08StyleNet/instance_normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2+
)StyleNet/instance_normalization/Reshape_1?
/StyleNet/instance_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:21
/StyleNet/instance_normalization/batchnorm/add/y?
-StyleNet/instance_normalization/batchnorm/addAddV29StyleNet/instance_normalization/moments/variance:output:08StyleNet/instance_normalization/batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????2/
-StyleNet/instance_normalization/batchnorm/add?
/StyleNet/instance_normalization/batchnorm/RsqrtRsqrt1StyleNet/instance_normalization/batchnorm/add:z:0*
T0*/
_output_shapes
:?????????21
/StyleNet/instance_normalization/batchnorm/Rsqrt?
-StyleNet/instance_normalization/batchnorm/mulMul3StyleNet/instance_normalization/batchnorm/Rsqrt:y:00StyleNet/instance_normalization/Reshape:output:0*
T0*/
_output_shapes
:?????????2/
-StyleNet/instance_normalization/batchnorm/mul?
/StyleNet/instance_normalization/batchnorm/mul_1Mul StyleNet/conv2d/BiasAdd:output:01StyleNet/instance_normalization/batchnorm/mul:z:0*
T0*1
_output_shapes
:???????????21
/StyleNet/instance_normalization/batchnorm/mul_1?
/StyleNet/instance_normalization/batchnorm/mul_2Mul5StyleNet/instance_normalization/moments/mean:output:01StyleNet/instance_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????21
/StyleNet/instance_normalization/batchnorm/mul_2?
-StyleNet/instance_normalization/batchnorm/subSub2StyleNet/instance_normalization/Reshape_1:output:03StyleNet/instance_normalization/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????2/
-StyleNet/instance_normalization/batchnorm/sub?
/StyleNet/instance_normalization/batchnorm/add_1AddV23StyleNet/instance_normalization/batchnorm/mul_1:z:01StyleNet/instance_normalization/batchnorm/sub:z:0*
T0*1
_output_shapes
:???????????21
/StyleNet/instance_normalization/batchnorm/add_1?
StyleNet/activation/ReluRelu3StyleNet/instance_normalization/batchnorm/add_1:z:0*
T0*1
_output_shapes
:???????????2
StyleNet/activation/Relu?
'StyleNet/conv2d_1/Conv2D/ReadVariableOpReadVariableOp0stylenet_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'StyleNet/conv2d_1/Conv2D/ReadVariableOp?
StyleNet/conv2d_1/Conv2DConv2D&StyleNet/activation/Relu:activations:0/StyleNet/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
StyleNet/conv2d_1/Conv2D?
(StyleNet/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp1stylenet_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(StyleNet/conv2d_1/BiasAdd/ReadVariableOp?
StyleNet/conv2d_1/BiasAddBiasAdd!StyleNet/conv2d_1/Conv2D:output:00StyleNet/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
StyleNet/conv2d_1/BiasAdd?
'StyleNet/instance_normalization_1/ShapeShape"StyleNet/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2)
'StyleNet/instance_normalization_1/Shape?
5StyleNet/instance_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5StyleNet/instance_normalization_1/strided_slice/stack?
7StyleNet/instance_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_1/strided_slice/stack_1?
7StyleNet/instance_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_1/strided_slice/stack_2?
/StyleNet/instance_normalization_1/strided_sliceStridedSlice0StyleNet/instance_normalization_1/Shape:output:0>StyleNet/instance_normalization_1/strided_slice/stack:output:0@StyleNet/instance_normalization_1/strided_slice/stack_1:output:0@StyleNet/instance_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/StyleNet/instance_normalization_1/strided_slice?
7StyleNet/instance_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_1/strided_slice_1/stack?
9StyleNet/instance_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_1/strided_slice_1/stack_1?
9StyleNet/instance_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_1/strided_slice_1/stack_2?
1StyleNet/instance_normalization_1/strided_slice_1StridedSlice0StyleNet/instance_normalization_1/Shape:output:0@StyleNet/instance_normalization_1/strided_slice_1/stack:output:0BStyleNet/instance_normalization_1/strided_slice_1/stack_1:output:0BStyleNet/instance_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1StyleNet/instance_normalization_1/strided_slice_1?
7StyleNet/instance_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_1/strided_slice_2/stack?
9StyleNet/instance_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_1/strided_slice_2/stack_1?
9StyleNet/instance_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_1/strided_slice_2/stack_2?
1StyleNet/instance_normalization_1/strided_slice_2StridedSlice0StyleNet/instance_normalization_1/Shape:output:0@StyleNet/instance_normalization_1/strided_slice_2/stack:output:0BStyleNet/instance_normalization_1/strided_slice_2/stack_1:output:0BStyleNet/instance_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1StyleNet/instance_normalization_1/strided_slice_2?
7StyleNet/instance_normalization_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_1/strided_slice_3/stack?
9StyleNet/instance_normalization_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_1/strided_slice_3/stack_1?
9StyleNet/instance_normalization_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_1/strided_slice_3/stack_2?
1StyleNet/instance_normalization_1/strided_slice_3StridedSlice0StyleNet/instance_normalization_1/Shape:output:0@StyleNet/instance_normalization_1/strided_slice_3/stack:output:0BStyleNet/instance_normalization_1/strided_slice_3/stack_1:output:0BStyleNet/instance_normalization_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1StyleNet/instance_normalization_1/strided_slice_3?
@StyleNet/instance_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2B
@StyleNet/instance_normalization_1/moments/mean/reduction_indices?
.StyleNet/instance_normalization_1/moments/meanMean"StyleNet/conv2d_1/BiasAdd:output:0IStyleNet/instance_normalization_1/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(20
.StyleNet/instance_normalization_1/moments/mean?
6StyleNet/instance_normalization_1/moments/StopGradientStopGradient7StyleNet/instance_normalization_1/moments/mean:output:0*
T0*/
_output_shapes
:????????? 28
6StyleNet/instance_normalization_1/moments/StopGradient?
;StyleNet/instance_normalization_1/moments/SquaredDifferenceSquaredDifference"StyleNet/conv2d_1/BiasAdd:output:0?StyleNet/instance_normalization_1/moments/StopGradient:output:0*
T0*1
_output_shapes
:??????????? 2=
;StyleNet/instance_normalization_1/moments/SquaredDifference?
DStyleNet/instance_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2F
DStyleNet/instance_normalization_1/moments/variance/reduction_indices?
2StyleNet/instance_normalization_1/moments/varianceMean?StyleNet/instance_normalization_1/moments/SquaredDifference:z:0MStyleNet/instance_normalization_1/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(24
2StyleNet/instance_normalization_1/moments/variance?
8StyleNet/instance_normalization_1/Reshape/ReadVariableOpReadVariableOpAstylenet_instance_normalization_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype02:
8StyleNet/instance_normalization_1/Reshape/ReadVariableOp?
/StyleNet/instance_normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             21
/StyleNet/instance_normalization_1/Reshape/shape?
)StyleNet/instance_normalization_1/ReshapeReshape@StyleNet/instance_normalization_1/Reshape/ReadVariableOp:value:08StyleNet/instance_normalization_1/Reshape/shape:output:0*
T0*&
_output_shapes
: 2+
)StyleNet/instance_normalization_1/Reshape?
:StyleNet/instance_normalization_1/Reshape_1/ReadVariableOpReadVariableOpCstylenet_instance_normalization_1_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02<
:StyleNet/instance_normalization_1/Reshape_1/ReadVariableOp?
1StyleNet/instance_normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             23
1StyleNet/instance_normalization_1/Reshape_1/shape?
+StyleNet/instance_normalization_1/Reshape_1ReshapeBStyleNet/instance_normalization_1/Reshape_1/ReadVariableOp:value:0:StyleNet/instance_normalization_1/Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2-
+StyleNet/instance_normalization_1/Reshape_1?
1StyleNet/instance_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:23
1StyleNet/instance_normalization_1/batchnorm/add/y?
/StyleNet/instance_normalization_1/batchnorm/addAddV2;StyleNet/instance_normalization_1/moments/variance:output:0:StyleNet/instance_normalization_1/batchnorm/add/y:output:0*
T0*/
_output_shapes
:????????? 21
/StyleNet/instance_normalization_1/batchnorm/add?
1StyleNet/instance_normalization_1/batchnorm/RsqrtRsqrt3StyleNet/instance_normalization_1/batchnorm/add:z:0*
T0*/
_output_shapes
:????????? 23
1StyleNet/instance_normalization_1/batchnorm/Rsqrt?
/StyleNet/instance_normalization_1/batchnorm/mulMul5StyleNet/instance_normalization_1/batchnorm/Rsqrt:y:02StyleNet/instance_normalization_1/Reshape:output:0*
T0*/
_output_shapes
:????????? 21
/StyleNet/instance_normalization_1/batchnorm/mul?
1StyleNet/instance_normalization_1/batchnorm/mul_1Mul"StyleNet/conv2d_1/BiasAdd:output:03StyleNet/instance_normalization_1/batchnorm/mul:z:0*
T0*1
_output_shapes
:??????????? 23
1StyleNet/instance_normalization_1/batchnorm/mul_1?
1StyleNet/instance_normalization_1/batchnorm/mul_2Mul7StyleNet/instance_normalization_1/moments/mean:output:03StyleNet/instance_normalization_1/batchnorm/mul:z:0*
T0*/
_output_shapes
:????????? 23
1StyleNet/instance_normalization_1/batchnorm/mul_2?
/StyleNet/instance_normalization_1/batchnorm/subSub4StyleNet/instance_normalization_1/Reshape_1:output:05StyleNet/instance_normalization_1/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:????????? 21
/StyleNet/instance_normalization_1/batchnorm/sub?
1StyleNet/instance_normalization_1/batchnorm/add_1AddV25StyleNet/instance_normalization_1/batchnorm/mul_1:z:03StyleNet/instance_normalization_1/batchnorm/sub:z:0*
T0*1
_output_shapes
:??????????? 23
1StyleNet/instance_normalization_1/batchnorm/add_1?
StyleNet/activation_1/ReluRelu5StyleNet/instance_normalization_1/batchnorm/add_1:z:0*
T0*1
_output_shapes
:??????????? 2
StyleNet/activation_1/Relu?
'StyleNet/conv2d_2/Conv2D/ReadVariableOpReadVariableOp0stylenet_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02)
'StyleNet/conv2d_2/Conv2D/ReadVariableOp?
StyleNet/conv2d_2/Conv2DConv2D(StyleNet/activation_1/Relu:activations:0/StyleNet/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
StyleNet/conv2d_2/Conv2D?
(StyleNet/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp1stylenet_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02*
(StyleNet/conv2d_2/BiasAdd/ReadVariableOp?
StyleNet/conv2d_2/BiasAddBiasAdd!StyleNet/conv2d_2/Conv2D:output:00StyleNet/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02
StyleNet/conv2d_2/BiasAdd?
'StyleNet/instance_normalization_2/ShapeShape"StyleNet/conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:2)
'StyleNet/instance_normalization_2/Shape?
5StyleNet/instance_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5StyleNet/instance_normalization_2/strided_slice/stack?
7StyleNet/instance_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_2/strided_slice/stack_1?
7StyleNet/instance_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_2/strided_slice/stack_2?
/StyleNet/instance_normalization_2/strided_sliceStridedSlice0StyleNet/instance_normalization_2/Shape:output:0>StyleNet/instance_normalization_2/strided_slice/stack:output:0@StyleNet/instance_normalization_2/strided_slice/stack_1:output:0@StyleNet/instance_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/StyleNet/instance_normalization_2/strided_slice?
7StyleNet/instance_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_2/strided_slice_1/stack?
9StyleNet/instance_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_2/strided_slice_1/stack_1?
9StyleNet/instance_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_2/strided_slice_1/stack_2?
1StyleNet/instance_normalization_2/strided_slice_1StridedSlice0StyleNet/instance_normalization_2/Shape:output:0@StyleNet/instance_normalization_2/strided_slice_1/stack:output:0BStyleNet/instance_normalization_2/strided_slice_1/stack_1:output:0BStyleNet/instance_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1StyleNet/instance_normalization_2/strided_slice_1?
7StyleNet/instance_normalization_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_2/strided_slice_2/stack?
9StyleNet/instance_normalization_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_2/strided_slice_2/stack_1?
9StyleNet/instance_normalization_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_2/strided_slice_2/stack_2?
1StyleNet/instance_normalization_2/strided_slice_2StridedSlice0StyleNet/instance_normalization_2/Shape:output:0@StyleNet/instance_normalization_2/strided_slice_2/stack:output:0BStyleNet/instance_normalization_2/strided_slice_2/stack_1:output:0BStyleNet/instance_normalization_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1StyleNet/instance_normalization_2/strided_slice_2?
7StyleNet/instance_normalization_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_2/strided_slice_3/stack?
9StyleNet/instance_normalization_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_2/strided_slice_3/stack_1?
9StyleNet/instance_normalization_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_2/strided_slice_3/stack_2?
1StyleNet/instance_normalization_2/strided_slice_3StridedSlice0StyleNet/instance_normalization_2/Shape:output:0@StyleNet/instance_normalization_2/strided_slice_3/stack:output:0BStyleNet/instance_normalization_2/strided_slice_3/stack_1:output:0BStyleNet/instance_normalization_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1StyleNet/instance_normalization_2/strided_slice_3?
@StyleNet/instance_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2B
@StyleNet/instance_normalization_2/moments/mean/reduction_indices?
.StyleNet/instance_normalization_2/moments/meanMean"StyleNet/conv2d_2/BiasAdd:output:0IStyleNet/instance_normalization_2/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(20
.StyleNet/instance_normalization_2/moments/mean?
6StyleNet/instance_normalization_2/moments/StopGradientStopGradient7StyleNet/instance_normalization_2/moments/mean:output:0*
T0*/
_output_shapes
:?????????028
6StyleNet/instance_normalization_2/moments/StopGradient?
;StyleNet/instance_normalization_2/moments/SquaredDifferenceSquaredDifference"StyleNet/conv2d_2/BiasAdd:output:0?StyleNet/instance_normalization_2/moments/StopGradient:output:0*
T0*0
_output_shapes
:?????????Z?02=
;StyleNet/instance_normalization_2/moments/SquaredDifference?
DStyleNet/instance_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2F
DStyleNet/instance_normalization_2/moments/variance/reduction_indices?
2StyleNet/instance_normalization_2/moments/varianceMean?StyleNet/instance_normalization_2/moments/SquaredDifference:z:0MStyleNet/instance_normalization_2/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(24
2StyleNet/instance_normalization_2/moments/variance?
8StyleNet/instance_normalization_2/Reshape/ReadVariableOpReadVariableOpAstylenet_instance_normalization_2_reshape_readvariableop_resource*
_output_shapes
:0*
dtype02:
8StyleNet/instance_normalization_2/Reshape/ReadVariableOp?
/StyleNet/instance_normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         0   21
/StyleNet/instance_normalization_2/Reshape/shape?
)StyleNet/instance_normalization_2/ReshapeReshape@StyleNet/instance_normalization_2/Reshape/ReadVariableOp:value:08StyleNet/instance_normalization_2/Reshape/shape:output:0*
T0*&
_output_shapes
:02+
)StyleNet/instance_normalization_2/Reshape?
:StyleNet/instance_normalization_2/Reshape_1/ReadVariableOpReadVariableOpCstylenet_instance_normalization_2_reshape_1_readvariableop_resource*
_output_shapes
:0*
dtype02<
:StyleNet/instance_normalization_2/Reshape_1/ReadVariableOp?
1StyleNet/instance_normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         0   23
1StyleNet/instance_normalization_2/Reshape_1/shape?
+StyleNet/instance_normalization_2/Reshape_1ReshapeBStyleNet/instance_normalization_2/Reshape_1/ReadVariableOp:value:0:StyleNet/instance_normalization_2/Reshape_1/shape:output:0*
T0*&
_output_shapes
:02-
+StyleNet/instance_normalization_2/Reshape_1?
1StyleNet/instance_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:23
1StyleNet/instance_normalization_2/batchnorm/add/y?
/StyleNet/instance_normalization_2/batchnorm/addAddV2;StyleNet/instance_normalization_2/moments/variance:output:0:StyleNet/instance_normalization_2/batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????021
/StyleNet/instance_normalization_2/batchnorm/add?
1StyleNet/instance_normalization_2/batchnorm/RsqrtRsqrt3StyleNet/instance_normalization_2/batchnorm/add:z:0*
T0*/
_output_shapes
:?????????023
1StyleNet/instance_normalization_2/batchnorm/Rsqrt?
/StyleNet/instance_normalization_2/batchnorm/mulMul5StyleNet/instance_normalization_2/batchnorm/Rsqrt:y:02StyleNet/instance_normalization_2/Reshape:output:0*
T0*/
_output_shapes
:?????????021
/StyleNet/instance_normalization_2/batchnorm/mul?
1StyleNet/instance_normalization_2/batchnorm/mul_1Mul"StyleNet/conv2d_2/BiasAdd:output:03StyleNet/instance_normalization_2/batchnorm/mul:z:0*
T0*0
_output_shapes
:?????????Z?023
1StyleNet/instance_normalization_2/batchnorm/mul_1?
1StyleNet/instance_normalization_2/batchnorm/mul_2Mul7StyleNet/instance_normalization_2/moments/mean:output:03StyleNet/instance_normalization_2/batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????023
1StyleNet/instance_normalization_2/batchnorm/mul_2?
/StyleNet/instance_normalization_2/batchnorm/subSub4StyleNet/instance_normalization_2/Reshape_1:output:05StyleNet/instance_normalization_2/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????021
/StyleNet/instance_normalization_2/batchnorm/sub?
1StyleNet/instance_normalization_2/batchnorm/add_1AddV25StyleNet/instance_normalization_2/batchnorm/mul_1:z:03StyleNet/instance_normalization_2/batchnorm/sub:z:0*
T0*0
_output_shapes
:?????????Z?023
1StyleNet/instance_normalization_2/batchnorm/add_1?
StyleNet/activation_2/ReluRelu5StyleNet/instance_normalization_2/batchnorm/add_1:z:0*
T0*0
_output_shapes
:?????????Z?02
StyleNet/activation_2/Relu?
6StyleNet/residual_block/conv2d_3/Conv2D/ReadVariableOpReadVariableOp?stylenet_residual_block_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype028
6StyleNet/residual_block/conv2d_3/Conv2D/ReadVariableOp?
'StyleNet/residual_block/conv2d_3/Conv2DConv2D(StyleNet/activation_2/Relu:activations:0>StyleNet/residual_block/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2)
'StyleNet/residual_block/conv2d_3/Conv2D?
7StyleNet/residual_block/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp@stylenet_residual_block_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype029
7StyleNet/residual_block/conv2d_3/BiasAdd/ReadVariableOp?
(StyleNet/residual_block/conv2d_3/BiasAddBiasAdd0StyleNet/residual_block/conv2d_3/Conv2D:output:0?StyleNet/residual_block/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02*
(StyleNet/residual_block/conv2d_3/BiasAdd?
%StyleNet/residual_block/conv2d_3/ReluRelu1StyleNet/residual_block/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02'
%StyleNet/residual_block/conv2d_3/Relu?
6StyleNet/residual_block/conv2d_4/Conv2D/ReadVariableOpReadVariableOp?stylenet_residual_block_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype028
6StyleNet/residual_block/conv2d_4/Conv2D/ReadVariableOp?
'StyleNet/residual_block/conv2d_4/Conv2DConv2D3StyleNet/residual_block/conv2d_3/Relu:activations:0>StyleNet/residual_block/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2)
'StyleNet/residual_block/conv2d_4/Conv2D?
7StyleNet/residual_block/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp@stylenet_residual_block_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype029
7StyleNet/residual_block/conv2d_4/BiasAdd/ReadVariableOp?
(StyleNet/residual_block/conv2d_4/BiasAddBiasAdd0StyleNet/residual_block/conv2d_4/Conv2D:output:0?StyleNet/residual_block/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02*
(StyleNet/residual_block/conv2d_4/BiasAdd?
StyleNet/residual_block/add/addAddV2(StyleNet/activation_2/Relu:activations:01StyleNet/residual_block/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02!
StyleNet/residual_block/add/add?
"StyleNet/residual_block/re_lu/ReluRelu#StyleNet/residual_block/add/add:z:0*
T0*0
_output_shapes
:?????????Z?02$
"StyleNet/residual_block/re_lu/Relu?
8StyleNet/residual_block_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOpAstylenet_residual_block_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02:
8StyleNet/residual_block_1/conv2d_5/Conv2D/ReadVariableOp?
)StyleNet/residual_block_1/conv2d_5/Conv2DConv2D0StyleNet/residual_block/re_lu/Relu:activations:0@StyleNet/residual_block_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2+
)StyleNet/residual_block_1/conv2d_5/Conv2D?
9StyleNet/residual_block_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpBstylenet_residual_block_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02;
9StyleNet/residual_block_1/conv2d_5/BiasAdd/ReadVariableOp?
*StyleNet/residual_block_1/conv2d_5/BiasAddBiasAdd2StyleNet/residual_block_1/conv2d_5/Conv2D:output:0AStyleNet/residual_block_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02,
*StyleNet/residual_block_1/conv2d_5/BiasAdd?
'StyleNet/residual_block_1/conv2d_5/ReluRelu3StyleNet/residual_block_1/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02)
'StyleNet/residual_block_1/conv2d_5/Relu?
8StyleNet/residual_block_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOpAstylenet_residual_block_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02:
8StyleNet/residual_block_1/conv2d_6/Conv2D/ReadVariableOp?
)StyleNet/residual_block_1/conv2d_6/Conv2DConv2D5StyleNet/residual_block_1/conv2d_5/Relu:activations:0@StyleNet/residual_block_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2+
)StyleNet/residual_block_1/conv2d_6/Conv2D?
9StyleNet/residual_block_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpBstylenet_residual_block_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02;
9StyleNet/residual_block_1/conv2d_6/BiasAdd/ReadVariableOp?
*StyleNet/residual_block_1/conv2d_6/BiasAddBiasAdd2StyleNet/residual_block_1/conv2d_6/Conv2D:output:0AStyleNet/residual_block_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02,
*StyleNet/residual_block_1/conv2d_6/BiasAdd?
#StyleNet/residual_block_1/add_1/addAddV20StyleNet/residual_block/re_lu/Relu:activations:03StyleNet/residual_block_1/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02%
#StyleNet/residual_block_1/add_1/add?
&StyleNet/residual_block_1/re_lu_1/ReluRelu'StyleNet/residual_block_1/add_1/add:z:0*
T0*0
_output_shapes
:?????????Z?02(
&StyleNet/residual_block_1/re_lu_1/Relu?
8StyleNet/residual_block_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOpAstylenet_residual_block_2_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02:
8StyleNet/residual_block_2/conv2d_7/Conv2D/ReadVariableOp?
)StyleNet/residual_block_2/conv2d_7/Conv2DConv2D4StyleNet/residual_block_1/re_lu_1/Relu:activations:0@StyleNet/residual_block_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2+
)StyleNet/residual_block_2/conv2d_7/Conv2D?
9StyleNet/residual_block_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOpBstylenet_residual_block_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02;
9StyleNet/residual_block_2/conv2d_7/BiasAdd/ReadVariableOp?
*StyleNet/residual_block_2/conv2d_7/BiasAddBiasAdd2StyleNet/residual_block_2/conv2d_7/Conv2D:output:0AStyleNet/residual_block_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02,
*StyleNet/residual_block_2/conv2d_7/BiasAdd?
'StyleNet/residual_block_2/conv2d_7/ReluRelu3StyleNet/residual_block_2/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02)
'StyleNet/residual_block_2/conv2d_7/Relu?
8StyleNet/residual_block_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOpAstylenet_residual_block_2_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02:
8StyleNet/residual_block_2/conv2d_8/Conv2D/ReadVariableOp?
)StyleNet/residual_block_2/conv2d_8/Conv2DConv2D5StyleNet/residual_block_2/conv2d_7/Relu:activations:0@StyleNet/residual_block_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2+
)StyleNet/residual_block_2/conv2d_8/Conv2D?
9StyleNet/residual_block_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOpBstylenet_residual_block_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02;
9StyleNet/residual_block_2/conv2d_8/BiasAdd/ReadVariableOp?
*StyleNet/residual_block_2/conv2d_8/BiasAddBiasAdd2StyleNet/residual_block_2/conv2d_8/Conv2D:output:0AStyleNet/residual_block_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02,
*StyleNet/residual_block_2/conv2d_8/BiasAdd?
#StyleNet/residual_block_2/add_2/addAddV24StyleNet/residual_block_1/re_lu_1/Relu:activations:03StyleNet/residual_block_2/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02%
#StyleNet/residual_block_2/add_2/add?
&StyleNet/residual_block_2/re_lu_2/ReluRelu'StyleNet/residual_block_2/add_2/add:z:0*
T0*0
_output_shapes
:?????????Z?02(
&StyleNet/residual_block_2/re_lu_2/Relu?
8StyleNet/residual_block_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOpAstylenet_residual_block_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02:
8StyleNet/residual_block_3/conv2d_9/Conv2D/ReadVariableOp?
)StyleNet/residual_block_3/conv2d_9/Conv2DConv2D4StyleNet/residual_block_2/re_lu_2/Relu:activations:0@StyleNet/residual_block_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2+
)StyleNet/residual_block_3/conv2d_9/Conv2D?
9StyleNet/residual_block_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOpBstylenet_residual_block_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02;
9StyleNet/residual_block_3/conv2d_9/BiasAdd/ReadVariableOp?
*StyleNet/residual_block_3/conv2d_9/BiasAddBiasAdd2StyleNet/residual_block_3/conv2d_9/Conv2D:output:0AStyleNet/residual_block_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02,
*StyleNet/residual_block_3/conv2d_9/BiasAdd?
'StyleNet/residual_block_3/conv2d_9/ReluRelu3StyleNet/residual_block_3/conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02)
'StyleNet/residual_block_3/conv2d_9/Relu?
9StyleNet/residual_block_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOpBstylenet_residual_block_3_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02;
9StyleNet/residual_block_3/conv2d_10/Conv2D/ReadVariableOp?
*StyleNet/residual_block_3/conv2d_10/Conv2DConv2D5StyleNet/residual_block_3/conv2d_9/Relu:activations:0AStyleNet/residual_block_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2,
*StyleNet/residual_block_3/conv2d_10/Conv2D?
:StyleNet/residual_block_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOpCstylenet_residual_block_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02<
:StyleNet/residual_block_3/conv2d_10/BiasAdd/ReadVariableOp?
+StyleNet/residual_block_3/conv2d_10/BiasAddBiasAdd3StyleNet/residual_block_3/conv2d_10/Conv2D:output:0BStyleNet/residual_block_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02-
+StyleNet/residual_block_3/conv2d_10/BiasAdd?
#StyleNet/residual_block_3/add_3/addAddV24StyleNet/residual_block_2/re_lu_2/Relu:activations:04StyleNet/residual_block_3/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02%
#StyleNet/residual_block_3/add_3/add?
&StyleNet/residual_block_3/re_lu_3/ReluRelu'StyleNet/residual_block_3/add_3/add:z:0*
T0*0
_output_shapes
:?????????Z?02(
&StyleNet/residual_block_3/re_lu_3/Relu?
9StyleNet/residual_block_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOpBstylenet_residual_block_4_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02;
9StyleNet/residual_block_4/conv2d_11/Conv2D/ReadVariableOp?
*StyleNet/residual_block_4/conv2d_11/Conv2DConv2D4StyleNet/residual_block_3/re_lu_3/Relu:activations:0AStyleNet/residual_block_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2,
*StyleNet/residual_block_4/conv2d_11/Conv2D?
:StyleNet/residual_block_4/conv2d_11/BiasAdd/ReadVariableOpReadVariableOpCstylenet_residual_block_4_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02<
:StyleNet/residual_block_4/conv2d_11/BiasAdd/ReadVariableOp?
+StyleNet/residual_block_4/conv2d_11/BiasAddBiasAdd3StyleNet/residual_block_4/conv2d_11/Conv2D:output:0BStyleNet/residual_block_4/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02-
+StyleNet/residual_block_4/conv2d_11/BiasAdd?
(StyleNet/residual_block_4/conv2d_11/ReluRelu4StyleNet/residual_block_4/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02*
(StyleNet/residual_block_4/conv2d_11/Relu?
9StyleNet/residual_block_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOpBstylenet_residual_block_4_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02;
9StyleNet/residual_block_4/conv2d_12/Conv2D/ReadVariableOp?
*StyleNet/residual_block_4/conv2d_12/Conv2DConv2D6StyleNet/residual_block_4/conv2d_11/Relu:activations:0AStyleNet/residual_block_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2,
*StyleNet/residual_block_4/conv2d_12/Conv2D?
:StyleNet/residual_block_4/conv2d_12/BiasAdd/ReadVariableOpReadVariableOpCstylenet_residual_block_4_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02<
:StyleNet/residual_block_4/conv2d_12/BiasAdd/ReadVariableOp?
+StyleNet/residual_block_4/conv2d_12/BiasAddBiasAdd3StyleNet/residual_block_4/conv2d_12/Conv2D:output:0BStyleNet/residual_block_4/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02-
+StyleNet/residual_block_4/conv2d_12/BiasAdd?
#StyleNet/residual_block_4/add_4/addAddV24StyleNet/residual_block_3/re_lu_3/Relu:activations:04StyleNet/residual_block_4/conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02%
#StyleNet/residual_block_4/add_4/add?
&StyleNet/residual_block_4/re_lu_4/ReluRelu'StyleNet/residual_block_4/add_4/add:z:0*
T0*0
_output_shapes
:?????????Z?02(
&StyleNet/residual_block_4/re_lu_4/Relu?
StyleNet/conv2d_transpose/ShapeShape4StyleNet/residual_block_4/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2!
StyleNet/conv2d_transpose/Shape?
-StyleNet/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-StyleNet/conv2d_transpose/strided_slice/stack?
/StyleNet/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/StyleNet/conv2d_transpose/strided_slice/stack_1?
/StyleNet/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/StyleNet/conv2d_transpose/strided_slice/stack_2?
'StyleNet/conv2d_transpose/strided_sliceStridedSlice(StyleNet/conv2d_transpose/Shape:output:06StyleNet/conv2d_transpose/strided_slice/stack:output:08StyleNet/conv2d_transpose/strided_slice/stack_1:output:08StyleNet/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'StyleNet/conv2d_transpose/strided_slice?
!StyleNet/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2#
!StyleNet/conv2d_transpose/stack/1?
!StyleNet/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2#
!StyleNet/conv2d_transpose/stack/2?
!StyleNet/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2#
!StyleNet/conv2d_transpose/stack/3?
StyleNet/conv2d_transpose/stackPack0StyleNet/conv2d_transpose/strided_slice:output:0*StyleNet/conv2d_transpose/stack/1:output:0*StyleNet/conv2d_transpose/stack/2:output:0*StyleNet/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2!
StyleNet/conv2d_transpose/stack?
/StyleNet/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/StyleNet/conv2d_transpose/strided_slice_1/stack?
1StyleNet/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1StyleNet/conv2d_transpose/strided_slice_1/stack_1?
1StyleNet/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1StyleNet/conv2d_transpose/strided_slice_1/stack_2?
)StyleNet/conv2d_transpose/strided_slice_1StridedSlice(StyleNet/conv2d_transpose/stack:output:08StyleNet/conv2d_transpose/strided_slice_1/stack:output:0:StyleNet/conv2d_transpose/strided_slice_1/stack_1:output:0:StyleNet/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)StyleNet/conv2d_transpose/strided_slice_1?
9StyleNet/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpBstylenet_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02;
9StyleNet/conv2d_transpose/conv2d_transpose/ReadVariableOp?
*StyleNet/conv2d_transpose/conv2d_transposeConv2DBackpropInput(StyleNet/conv2d_transpose/stack:output:0AStyleNet/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:04StyleNet/residual_block_4/re_lu_4/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2,
*StyleNet/conv2d_transpose/conv2d_transpose?
0StyleNet/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp9stylenet_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0StyleNet/conv2d_transpose/BiasAdd/ReadVariableOp?
!StyleNet/conv2d_transpose/BiasAddBiasAdd3StyleNet/conv2d_transpose/conv2d_transpose:output:08StyleNet/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2#
!StyleNet/conv2d_transpose/BiasAdd?
'StyleNet/instance_normalization_3/ShapeShape*StyleNet/conv2d_transpose/BiasAdd:output:0*
T0*
_output_shapes
:2)
'StyleNet/instance_normalization_3/Shape?
5StyleNet/instance_normalization_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5StyleNet/instance_normalization_3/strided_slice/stack?
7StyleNet/instance_normalization_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_3/strided_slice/stack_1?
7StyleNet/instance_normalization_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_3/strided_slice/stack_2?
/StyleNet/instance_normalization_3/strided_sliceStridedSlice0StyleNet/instance_normalization_3/Shape:output:0>StyleNet/instance_normalization_3/strided_slice/stack:output:0@StyleNet/instance_normalization_3/strided_slice/stack_1:output:0@StyleNet/instance_normalization_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/StyleNet/instance_normalization_3/strided_slice?
7StyleNet/instance_normalization_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_3/strided_slice_1/stack?
9StyleNet/instance_normalization_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_3/strided_slice_1/stack_1?
9StyleNet/instance_normalization_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_3/strided_slice_1/stack_2?
1StyleNet/instance_normalization_3/strided_slice_1StridedSlice0StyleNet/instance_normalization_3/Shape:output:0@StyleNet/instance_normalization_3/strided_slice_1/stack:output:0BStyleNet/instance_normalization_3/strided_slice_1/stack_1:output:0BStyleNet/instance_normalization_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1StyleNet/instance_normalization_3/strided_slice_1?
7StyleNet/instance_normalization_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_3/strided_slice_2/stack?
9StyleNet/instance_normalization_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_3/strided_slice_2/stack_1?
9StyleNet/instance_normalization_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_3/strided_slice_2/stack_2?
1StyleNet/instance_normalization_3/strided_slice_2StridedSlice0StyleNet/instance_normalization_3/Shape:output:0@StyleNet/instance_normalization_3/strided_slice_2/stack:output:0BStyleNet/instance_normalization_3/strided_slice_2/stack_1:output:0BStyleNet/instance_normalization_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1StyleNet/instance_normalization_3/strided_slice_2?
7StyleNet/instance_normalization_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_3/strided_slice_3/stack?
9StyleNet/instance_normalization_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_3/strided_slice_3/stack_1?
9StyleNet/instance_normalization_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_3/strided_slice_3/stack_2?
1StyleNet/instance_normalization_3/strided_slice_3StridedSlice0StyleNet/instance_normalization_3/Shape:output:0@StyleNet/instance_normalization_3/strided_slice_3/stack:output:0BStyleNet/instance_normalization_3/strided_slice_3/stack_1:output:0BStyleNet/instance_normalization_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1StyleNet/instance_normalization_3/strided_slice_3?
@StyleNet/instance_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2B
@StyleNet/instance_normalization_3/moments/mean/reduction_indices?
.StyleNet/instance_normalization_3/moments/meanMean*StyleNet/conv2d_transpose/BiasAdd:output:0IStyleNet/instance_normalization_3/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(20
.StyleNet/instance_normalization_3/moments/mean?
6StyleNet/instance_normalization_3/moments/StopGradientStopGradient7StyleNet/instance_normalization_3/moments/mean:output:0*
T0*/
_output_shapes
:????????? 28
6StyleNet/instance_normalization_3/moments/StopGradient?
;StyleNet/instance_normalization_3/moments/SquaredDifferenceSquaredDifference*StyleNet/conv2d_transpose/BiasAdd:output:0?StyleNet/instance_normalization_3/moments/StopGradient:output:0*
T0*1
_output_shapes
:??????????? 2=
;StyleNet/instance_normalization_3/moments/SquaredDifference?
DStyleNet/instance_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2F
DStyleNet/instance_normalization_3/moments/variance/reduction_indices?
2StyleNet/instance_normalization_3/moments/varianceMean?StyleNet/instance_normalization_3/moments/SquaredDifference:z:0MStyleNet/instance_normalization_3/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(24
2StyleNet/instance_normalization_3/moments/variance?
8StyleNet/instance_normalization_3/Reshape/ReadVariableOpReadVariableOpAstylenet_instance_normalization_3_reshape_readvariableop_resource*
_output_shapes
: *
dtype02:
8StyleNet/instance_normalization_3/Reshape/ReadVariableOp?
/StyleNet/instance_normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             21
/StyleNet/instance_normalization_3/Reshape/shape?
)StyleNet/instance_normalization_3/ReshapeReshape@StyleNet/instance_normalization_3/Reshape/ReadVariableOp:value:08StyleNet/instance_normalization_3/Reshape/shape:output:0*
T0*&
_output_shapes
: 2+
)StyleNet/instance_normalization_3/Reshape?
:StyleNet/instance_normalization_3/Reshape_1/ReadVariableOpReadVariableOpCstylenet_instance_normalization_3_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02<
:StyleNet/instance_normalization_3/Reshape_1/ReadVariableOp?
1StyleNet/instance_normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             23
1StyleNet/instance_normalization_3/Reshape_1/shape?
+StyleNet/instance_normalization_3/Reshape_1ReshapeBStyleNet/instance_normalization_3/Reshape_1/ReadVariableOp:value:0:StyleNet/instance_normalization_3/Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2-
+StyleNet/instance_normalization_3/Reshape_1?
1StyleNet/instance_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:23
1StyleNet/instance_normalization_3/batchnorm/add/y?
/StyleNet/instance_normalization_3/batchnorm/addAddV2;StyleNet/instance_normalization_3/moments/variance:output:0:StyleNet/instance_normalization_3/batchnorm/add/y:output:0*
T0*/
_output_shapes
:????????? 21
/StyleNet/instance_normalization_3/batchnorm/add?
1StyleNet/instance_normalization_3/batchnorm/RsqrtRsqrt3StyleNet/instance_normalization_3/batchnorm/add:z:0*
T0*/
_output_shapes
:????????? 23
1StyleNet/instance_normalization_3/batchnorm/Rsqrt?
/StyleNet/instance_normalization_3/batchnorm/mulMul5StyleNet/instance_normalization_3/batchnorm/Rsqrt:y:02StyleNet/instance_normalization_3/Reshape:output:0*
T0*/
_output_shapes
:????????? 21
/StyleNet/instance_normalization_3/batchnorm/mul?
1StyleNet/instance_normalization_3/batchnorm/mul_1Mul*StyleNet/conv2d_transpose/BiasAdd:output:03StyleNet/instance_normalization_3/batchnorm/mul:z:0*
T0*1
_output_shapes
:??????????? 23
1StyleNet/instance_normalization_3/batchnorm/mul_1?
1StyleNet/instance_normalization_3/batchnorm/mul_2Mul7StyleNet/instance_normalization_3/moments/mean:output:03StyleNet/instance_normalization_3/batchnorm/mul:z:0*
T0*/
_output_shapes
:????????? 23
1StyleNet/instance_normalization_3/batchnorm/mul_2?
/StyleNet/instance_normalization_3/batchnorm/subSub4StyleNet/instance_normalization_3/Reshape_1:output:05StyleNet/instance_normalization_3/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:????????? 21
/StyleNet/instance_normalization_3/batchnorm/sub?
1StyleNet/instance_normalization_3/batchnorm/add_1AddV25StyleNet/instance_normalization_3/batchnorm/mul_1:z:03StyleNet/instance_normalization_3/batchnorm/sub:z:0*
T0*1
_output_shapes
:??????????? 23
1StyleNet/instance_normalization_3/batchnorm/add_1?
StyleNet/activation_3/ReluRelu5StyleNet/instance_normalization_3/batchnorm/add_1:z:0*
T0*1
_output_shapes
:??????????? 2
StyleNet/activation_3/Relu?
!StyleNet/conv2d_transpose_1/ShapeShape(StyleNet/activation_3/Relu:activations:0*
T0*
_output_shapes
:2#
!StyleNet/conv2d_transpose_1/Shape?
/StyleNet/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/StyleNet/conv2d_transpose_1/strided_slice/stack?
1StyleNet/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1StyleNet/conv2d_transpose_1/strided_slice/stack_1?
1StyleNet/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1StyleNet/conv2d_transpose_1/strided_slice/stack_2?
)StyleNet/conv2d_transpose_1/strided_sliceStridedSlice*StyleNet/conv2d_transpose_1/Shape:output:08StyleNet/conv2d_transpose_1/strided_slice/stack:output:0:StyleNet/conv2d_transpose_1/strided_slice/stack_1:output:0:StyleNet/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)StyleNet/conv2d_transpose_1/strided_slice?
#StyleNet/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#StyleNet/conv2d_transpose_1/stack/1?
#StyleNet/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2%
#StyleNet/conv2d_transpose_1/stack/2?
#StyleNet/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#StyleNet/conv2d_transpose_1/stack/3?
!StyleNet/conv2d_transpose_1/stackPack2StyleNet/conv2d_transpose_1/strided_slice:output:0,StyleNet/conv2d_transpose_1/stack/1:output:0,StyleNet/conv2d_transpose_1/stack/2:output:0,StyleNet/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!StyleNet/conv2d_transpose_1/stack?
1StyleNet/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1StyleNet/conv2d_transpose_1/strided_slice_1/stack?
3StyleNet/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3StyleNet/conv2d_transpose_1/strided_slice_1/stack_1?
3StyleNet/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3StyleNet/conv2d_transpose_1/strided_slice_1/stack_2?
+StyleNet/conv2d_transpose_1/strided_slice_1StridedSlice*StyleNet/conv2d_transpose_1/stack:output:0:StyleNet/conv2d_transpose_1/strided_slice_1/stack:output:0<StyleNet/conv2d_transpose_1/strided_slice_1/stack_1:output:0<StyleNet/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+StyleNet/conv2d_transpose_1/strided_slice_1?
;StyleNet/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpDstylenet_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02=
;StyleNet/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
,StyleNet/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput*StyleNet/conv2d_transpose_1/stack:output:0CStyleNet/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0(StyleNet/activation_3/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2.
,StyleNet/conv2d_transpose_1/conv2d_transpose?
2StyleNet/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp;stylenet_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2StyleNet/conv2d_transpose_1/BiasAdd/ReadVariableOp?
#StyleNet/conv2d_transpose_1/BiasAddBiasAdd5StyleNet/conv2d_transpose_1/conv2d_transpose:output:0:StyleNet/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2%
#StyleNet/conv2d_transpose_1/BiasAdd?
'StyleNet/instance_normalization_4/ShapeShape,StyleNet/conv2d_transpose_1/BiasAdd:output:0*
T0*
_output_shapes
:2)
'StyleNet/instance_normalization_4/Shape?
5StyleNet/instance_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5StyleNet/instance_normalization_4/strided_slice/stack?
7StyleNet/instance_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_4/strided_slice/stack_1?
7StyleNet/instance_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_4/strided_slice/stack_2?
/StyleNet/instance_normalization_4/strided_sliceStridedSlice0StyleNet/instance_normalization_4/Shape:output:0>StyleNet/instance_normalization_4/strided_slice/stack:output:0@StyleNet/instance_normalization_4/strided_slice/stack_1:output:0@StyleNet/instance_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/StyleNet/instance_normalization_4/strided_slice?
7StyleNet/instance_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_4/strided_slice_1/stack?
9StyleNet/instance_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_4/strided_slice_1/stack_1?
9StyleNet/instance_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_4/strided_slice_1/stack_2?
1StyleNet/instance_normalization_4/strided_slice_1StridedSlice0StyleNet/instance_normalization_4/Shape:output:0@StyleNet/instance_normalization_4/strided_slice_1/stack:output:0BStyleNet/instance_normalization_4/strided_slice_1/stack_1:output:0BStyleNet/instance_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1StyleNet/instance_normalization_4/strided_slice_1?
7StyleNet/instance_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_4/strided_slice_2/stack?
9StyleNet/instance_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_4/strided_slice_2/stack_1?
9StyleNet/instance_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_4/strided_slice_2/stack_2?
1StyleNet/instance_normalization_4/strided_slice_2StridedSlice0StyleNet/instance_normalization_4/Shape:output:0@StyleNet/instance_normalization_4/strided_slice_2/stack:output:0BStyleNet/instance_normalization_4/strided_slice_2/stack_1:output:0BStyleNet/instance_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1StyleNet/instance_normalization_4/strided_slice_2?
7StyleNet/instance_normalization_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_4/strided_slice_3/stack?
9StyleNet/instance_normalization_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_4/strided_slice_3/stack_1?
9StyleNet/instance_normalization_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_4/strided_slice_3/stack_2?
1StyleNet/instance_normalization_4/strided_slice_3StridedSlice0StyleNet/instance_normalization_4/Shape:output:0@StyleNet/instance_normalization_4/strided_slice_3/stack:output:0BStyleNet/instance_normalization_4/strided_slice_3/stack_1:output:0BStyleNet/instance_normalization_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1StyleNet/instance_normalization_4/strided_slice_3?
@StyleNet/instance_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2B
@StyleNet/instance_normalization_4/moments/mean/reduction_indices?
.StyleNet/instance_normalization_4/moments/meanMean,StyleNet/conv2d_transpose_1/BiasAdd:output:0IStyleNet/instance_normalization_4/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(20
.StyleNet/instance_normalization_4/moments/mean?
6StyleNet/instance_normalization_4/moments/StopGradientStopGradient7StyleNet/instance_normalization_4/moments/mean:output:0*
T0*/
_output_shapes
:?????????28
6StyleNet/instance_normalization_4/moments/StopGradient?
;StyleNet/instance_normalization_4/moments/SquaredDifferenceSquaredDifference,StyleNet/conv2d_transpose_1/BiasAdd:output:0?StyleNet/instance_normalization_4/moments/StopGradient:output:0*
T0*1
_output_shapes
:???????????2=
;StyleNet/instance_normalization_4/moments/SquaredDifference?
DStyleNet/instance_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2F
DStyleNet/instance_normalization_4/moments/variance/reduction_indices?
2StyleNet/instance_normalization_4/moments/varianceMean?StyleNet/instance_normalization_4/moments/SquaredDifference:z:0MStyleNet/instance_normalization_4/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(24
2StyleNet/instance_normalization_4/moments/variance?
8StyleNet/instance_normalization_4/Reshape/ReadVariableOpReadVariableOpAstylenet_instance_normalization_4_reshape_readvariableop_resource*
_output_shapes
:*
dtype02:
8StyleNet/instance_normalization_4/Reshape/ReadVariableOp?
/StyleNet/instance_normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/StyleNet/instance_normalization_4/Reshape/shape?
)StyleNet/instance_normalization_4/ReshapeReshape@StyleNet/instance_normalization_4/Reshape/ReadVariableOp:value:08StyleNet/instance_normalization_4/Reshape/shape:output:0*
T0*&
_output_shapes
:2+
)StyleNet/instance_normalization_4/Reshape?
:StyleNet/instance_normalization_4/Reshape_1/ReadVariableOpReadVariableOpCstylenet_instance_normalization_4_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02<
:StyleNet/instance_normalization_4/Reshape_1/ReadVariableOp?
1StyleNet/instance_normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            23
1StyleNet/instance_normalization_4/Reshape_1/shape?
+StyleNet/instance_normalization_4/Reshape_1ReshapeBStyleNet/instance_normalization_4/Reshape_1/ReadVariableOp:value:0:StyleNet/instance_normalization_4/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2-
+StyleNet/instance_normalization_4/Reshape_1?
1StyleNet/instance_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:23
1StyleNet/instance_normalization_4/batchnorm/add/y?
/StyleNet/instance_normalization_4/batchnorm/addAddV2;StyleNet/instance_normalization_4/moments/variance:output:0:StyleNet/instance_normalization_4/batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????21
/StyleNet/instance_normalization_4/batchnorm/add?
1StyleNet/instance_normalization_4/batchnorm/RsqrtRsqrt3StyleNet/instance_normalization_4/batchnorm/add:z:0*
T0*/
_output_shapes
:?????????23
1StyleNet/instance_normalization_4/batchnorm/Rsqrt?
/StyleNet/instance_normalization_4/batchnorm/mulMul5StyleNet/instance_normalization_4/batchnorm/Rsqrt:y:02StyleNet/instance_normalization_4/Reshape:output:0*
T0*/
_output_shapes
:?????????21
/StyleNet/instance_normalization_4/batchnorm/mul?
1StyleNet/instance_normalization_4/batchnorm/mul_1Mul,StyleNet/conv2d_transpose_1/BiasAdd:output:03StyleNet/instance_normalization_4/batchnorm/mul:z:0*
T0*1
_output_shapes
:???????????23
1StyleNet/instance_normalization_4/batchnorm/mul_1?
1StyleNet/instance_normalization_4/batchnorm/mul_2Mul7StyleNet/instance_normalization_4/moments/mean:output:03StyleNet/instance_normalization_4/batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????23
1StyleNet/instance_normalization_4/batchnorm/mul_2?
/StyleNet/instance_normalization_4/batchnorm/subSub4StyleNet/instance_normalization_4/Reshape_1:output:05StyleNet/instance_normalization_4/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????21
/StyleNet/instance_normalization_4/batchnorm/sub?
1StyleNet/instance_normalization_4/batchnorm/add_1AddV25StyleNet/instance_normalization_4/batchnorm/mul_1:z:03StyleNet/instance_normalization_4/batchnorm/sub:z:0*
T0*1
_output_shapes
:???????????23
1StyleNet/instance_normalization_4/batchnorm/add_1?
StyleNet/activation_4/ReluRelu5StyleNet/instance_normalization_4/batchnorm/add_1:z:0*
T0*1
_output_shapes
:???????????2
StyleNet/activation_4/Relu?
(StyleNet/conv2d_13/Conv2D/ReadVariableOpReadVariableOp1stylenet_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(StyleNet/conv2d_13/Conv2D/ReadVariableOp?
StyleNet/conv2d_13/Conv2DConv2D(StyleNet/activation_4/Relu:activations:00StyleNet/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
StyleNet/conv2d_13/Conv2D?
)StyleNet/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp2stylenet_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)StyleNet/conv2d_13/BiasAdd/ReadVariableOp?
StyleNet/conv2d_13/BiasAddBiasAdd"StyleNet/conv2d_13/Conv2D:output:01StyleNet/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
StyleNet/conv2d_13/BiasAdd?
'StyleNet/instance_normalization_5/ShapeShape#StyleNet/conv2d_13/BiasAdd:output:0*
T0*
_output_shapes
:2)
'StyleNet/instance_normalization_5/Shape?
5StyleNet/instance_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5StyleNet/instance_normalization_5/strided_slice/stack?
7StyleNet/instance_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_5/strided_slice/stack_1?
7StyleNet/instance_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_5/strided_slice/stack_2?
/StyleNet/instance_normalization_5/strided_sliceStridedSlice0StyleNet/instance_normalization_5/Shape:output:0>StyleNet/instance_normalization_5/strided_slice/stack:output:0@StyleNet/instance_normalization_5/strided_slice/stack_1:output:0@StyleNet/instance_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/StyleNet/instance_normalization_5/strided_slice?
7StyleNet/instance_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_5/strided_slice_1/stack?
9StyleNet/instance_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_5/strided_slice_1/stack_1?
9StyleNet/instance_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_5/strided_slice_1/stack_2?
1StyleNet/instance_normalization_5/strided_slice_1StridedSlice0StyleNet/instance_normalization_5/Shape:output:0@StyleNet/instance_normalization_5/strided_slice_1/stack:output:0BStyleNet/instance_normalization_5/strided_slice_1/stack_1:output:0BStyleNet/instance_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1StyleNet/instance_normalization_5/strided_slice_1?
7StyleNet/instance_normalization_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_5/strided_slice_2/stack?
9StyleNet/instance_normalization_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_5/strided_slice_2/stack_1?
9StyleNet/instance_normalization_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_5/strided_slice_2/stack_2?
1StyleNet/instance_normalization_5/strided_slice_2StridedSlice0StyleNet/instance_normalization_5/Shape:output:0@StyleNet/instance_normalization_5/strided_slice_2/stack:output:0BStyleNet/instance_normalization_5/strided_slice_2/stack_1:output:0BStyleNet/instance_normalization_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1StyleNet/instance_normalization_5/strided_slice_2?
7StyleNet/instance_normalization_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7StyleNet/instance_normalization_5/strided_slice_3/stack?
9StyleNet/instance_normalization_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_5/strided_slice_3/stack_1?
9StyleNet/instance_normalization_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9StyleNet/instance_normalization_5/strided_slice_3/stack_2?
1StyleNet/instance_normalization_5/strided_slice_3StridedSlice0StyleNet/instance_normalization_5/Shape:output:0@StyleNet/instance_normalization_5/strided_slice_3/stack:output:0BStyleNet/instance_normalization_5/strided_slice_3/stack_1:output:0BStyleNet/instance_normalization_5/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1StyleNet/instance_normalization_5/strided_slice_3?
@StyleNet/instance_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2B
@StyleNet/instance_normalization_5/moments/mean/reduction_indices?
.StyleNet/instance_normalization_5/moments/meanMean#StyleNet/conv2d_13/BiasAdd:output:0IStyleNet/instance_normalization_5/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(20
.StyleNet/instance_normalization_5/moments/mean?
6StyleNet/instance_normalization_5/moments/StopGradientStopGradient7StyleNet/instance_normalization_5/moments/mean:output:0*
T0*/
_output_shapes
:?????????28
6StyleNet/instance_normalization_5/moments/StopGradient?
;StyleNet/instance_normalization_5/moments/SquaredDifferenceSquaredDifference#StyleNet/conv2d_13/BiasAdd:output:0?StyleNet/instance_normalization_5/moments/StopGradient:output:0*
T0*1
_output_shapes
:???????????2=
;StyleNet/instance_normalization_5/moments/SquaredDifference?
DStyleNet/instance_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2F
DStyleNet/instance_normalization_5/moments/variance/reduction_indices?
2StyleNet/instance_normalization_5/moments/varianceMean?StyleNet/instance_normalization_5/moments/SquaredDifference:z:0MStyleNet/instance_normalization_5/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(24
2StyleNet/instance_normalization_5/moments/variance?
8StyleNet/instance_normalization_5/Reshape/ReadVariableOpReadVariableOpAstylenet_instance_normalization_5_reshape_readvariableop_resource*
_output_shapes
:*
dtype02:
8StyleNet/instance_normalization_5/Reshape/ReadVariableOp?
/StyleNet/instance_normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/StyleNet/instance_normalization_5/Reshape/shape?
)StyleNet/instance_normalization_5/ReshapeReshape@StyleNet/instance_normalization_5/Reshape/ReadVariableOp:value:08StyleNet/instance_normalization_5/Reshape/shape:output:0*
T0*&
_output_shapes
:2+
)StyleNet/instance_normalization_5/Reshape?
:StyleNet/instance_normalization_5/Reshape_1/ReadVariableOpReadVariableOpCstylenet_instance_normalization_5_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02<
:StyleNet/instance_normalization_5/Reshape_1/ReadVariableOp?
1StyleNet/instance_normalization_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            23
1StyleNet/instance_normalization_5/Reshape_1/shape?
+StyleNet/instance_normalization_5/Reshape_1ReshapeBStyleNet/instance_normalization_5/Reshape_1/ReadVariableOp:value:0:StyleNet/instance_normalization_5/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2-
+StyleNet/instance_normalization_5/Reshape_1?
1StyleNet/instance_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:23
1StyleNet/instance_normalization_5/batchnorm/add/y?
/StyleNet/instance_normalization_5/batchnorm/addAddV2;StyleNet/instance_normalization_5/moments/variance:output:0:StyleNet/instance_normalization_5/batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????21
/StyleNet/instance_normalization_5/batchnorm/add?
1StyleNet/instance_normalization_5/batchnorm/RsqrtRsqrt3StyleNet/instance_normalization_5/batchnorm/add:z:0*
T0*/
_output_shapes
:?????????23
1StyleNet/instance_normalization_5/batchnorm/Rsqrt?
/StyleNet/instance_normalization_5/batchnorm/mulMul5StyleNet/instance_normalization_5/batchnorm/Rsqrt:y:02StyleNet/instance_normalization_5/Reshape:output:0*
T0*/
_output_shapes
:?????????21
/StyleNet/instance_normalization_5/batchnorm/mul?
1StyleNet/instance_normalization_5/batchnorm/mul_1Mul#StyleNet/conv2d_13/BiasAdd:output:03StyleNet/instance_normalization_5/batchnorm/mul:z:0*
T0*1
_output_shapes
:???????????23
1StyleNet/instance_normalization_5/batchnorm/mul_1?
1StyleNet/instance_normalization_5/batchnorm/mul_2Mul7StyleNet/instance_normalization_5/moments/mean:output:03StyleNet/instance_normalization_5/batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????23
1StyleNet/instance_normalization_5/batchnorm/mul_2?
/StyleNet/instance_normalization_5/batchnorm/subSub4StyleNet/instance_normalization_5/Reshape_1:output:05StyleNet/instance_normalization_5/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????21
/StyleNet/instance_normalization_5/batchnorm/sub?
1StyleNet/instance_normalization_5/batchnorm/add_1AddV25StyleNet/instance_normalization_5/batchnorm/mul_1:z:03StyleNet/instance_normalization_5/batchnorm/sub:z:0*
T0*1
_output_shapes
:???????????23
1StyleNet/instance_normalization_5/batchnorm/add_1?
StyleNet/activation_5/TanhTanh5StyleNet/instance_normalization_5/batchnorm/add_1:z:0*
T0*1
_output_shapes
:???????????2
StyleNet/activation_5/Tanh?
IdentityIdentityStyleNet/activation_5/Tanh:y:0'^StyleNet/conv2d/BiasAdd/ReadVariableOp&^StyleNet/conv2d/Conv2D/ReadVariableOp)^StyleNet/conv2d_1/BiasAdd/ReadVariableOp(^StyleNet/conv2d_1/Conv2D/ReadVariableOp*^StyleNet/conv2d_13/BiasAdd/ReadVariableOp)^StyleNet/conv2d_13/Conv2D/ReadVariableOp)^StyleNet/conv2d_2/BiasAdd/ReadVariableOp(^StyleNet/conv2d_2/Conv2D/ReadVariableOp1^StyleNet/conv2d_transpose/BiasAdd/ReadVariableOp:^StyleNet/conv2d_transpose/conv2d_transpose/ReadVariableOp3^StyleNet/conv2d_transpose_1/BiasAdd/ReadVariableOp<^StyleNet/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^StyleNet/instance_normalization/Reshape/ReadVariableOp9^StyleNet/instance_normalization/Reshape_1/ReadVariableOp9^StyleNet/instance_normalization_1/Reshape/ReadVariableOp;^StyleNet/instance_normalization_1/Reshape_1/ReadVariableOp9^StyleNet/instance_normalization_2/Reshape/ReadVariableOp;^StyleNet/instance_normalization_2/Reshape_1/ReadVariableOp9^StyleNet/instance_normalization_3/Reshape/ReadVariableOp;^StyleNet/instance_normalization_3/Reshape_1/ReadVariableOp9^StyleNet/instance_normalization_4/Reshape/ReadVariableOp;^StyleNet/instance_normalization_4/Reshape_1/ReadVariableOp9^StyleNet/instance_normalization_5/Reshape/ReadVariableOp;^StyleNet/instance_normalization_5/Reshape_1/ReadVariableOp8^StyleNet/residual_block/conv2d_3/BiasAdd/ReadVariableOp7^StyleNet/residual_block/conv2d_3/Conv2D/ReadVariableOp8^StyleNet/residual_block/conv2d_4/BiasAdd/ReadVariableOp7^StyleNet/residual_block/conv2d_4/Conv2D/ReadVariableOp:^StyleNet/residual_block_1/conv2d_5/BiasAdd/ReadVariableOp9^StyleNet/residual_block_1/conv2d_5/Conv2D/ReadVariableOp:^StyleNet/residual_block_1/conv2d_6/BiasAdd/ReadVariableOp9^StyleNet/residual_block_1/conv2d_6/Conv2D/ReadVariableOp:^StyleNet/residual_block_2/conv2d_7/BiasAdd/ReadVariableOp9^StyleNet/residual_block_2/conv2d_7/Conv2D/ReadVariableOp:^StyleNet/residual_block_2/conv2d_8/BiasAdd/ReadVariableOp9^StyleNet/residual_block_2/conv2d_8/Conv2D/ReadVariableOp;^StyleNet/residual_block_3/conv2d_10/BiasAdd/ReadVariableOp:^StyleNet/residual_block_3/conv2d_10/Conv2D/ReadVariableOp:^StyleNet/residual_block_3/conv2d_9/BiasAdd/ReadVariableOp9^StyleNet/residual_block_3/conv2d_9/Conv2D/ReadVariableOp;^StyleNet/residual_block_4/conv2d_11/BiasAdd/ReadVariableOp:^StyleNet/residual_block_4/conv2d_11/Conv2D/ReadVariableOp;^StyleNet/residual_block_4/conv2d_12/BiasAdd/ReadVariableOp:^StyleNet/residual_block_4/conv2d_12/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&StyleNet/conv2d/BiasAdd/ReadVariableOp&StyleNet/conv2d/BiasAdd/ReadVariableOp2N
%StyleNet/conv2d/Conv2D/ReadVariableOp%StyleNet/conv2d/Conv2D/ReadVariableOp2T
(StyleNet/conv2d_1/BiasAdd/ReadVariableOp(StyleNet/conv2d_1/BiasAdd/ReadVariableOp2R
'StyleNet/conv2d_1/Conv2D/ReadVariableOp'StyleNet/conv2d_1/Conv2D/ReadVariableOp2V
)StyleNet/conv2d_13/BiasAdd/ReadVariableOp)StyleNet/conv2d_13/BiasAdd/ReadVariableOp2T
(StyleNet/conv2d_13/Conv2D/ReadVariableOp(StyleNet/conv2d_13/Conv2D/ReadVariableOp2T
(StyleNet/conv2d_2/BiasAdd/ReadVariableOp(StyleNet/conv2d_2/BiasAdd/ReadVariableOp2R
'StyleNet/conv2d_2/Conv2D/ReadVariableOp'StyleNet/conv2d_2/Conv2D/ReadVariableOp2d
0StyleNet/conv2d_transpose/BiasAdd/ReadVariableOp0StyleNet/conv2d_transpose/BiasAdd/ReadVariableOp2v
9StyleNet/conv2d_transpose/conv2d_transpose/ReadVariableOp9StyleNet/conv2d_transpose/conv2d_transpose/ReadVariableOp2h
2StyleNet/conv2d_transpose_1/BiasAdd/ReadVariableOp2StyleNet/conv2d_transpose_1/BiasAdd/ReadVariableOp2z
;StyleNet/conv2d_transpose_1/conv2d_transpose/ReadVariableOp;StyleNet/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6StyleNet/instance_normalization/Reshape/ReadVariableOp6StyleNet/instance_normalization/Reshape/ReadVariableOp2t
8StyleNet/instance_normalization/Reshape_1/ReadVariableOp8StyleNet/instance_normalization/Reshape_1/ReadVariableOp2t
8StyleNet/instance_normalization_1/Reshape/ReadVariableOp8StyleNet/instance_normalization_1/Reshape/ReadVariableOp2x
:StyleNet/instance_normalization_1/Reshape_1/ReadVariableOp:StyleNet/instance_normalization_1/Reshape_1/ReadVariableOp2t
8StyleNet/instance_normalization_2/Reshape/ReadVariableOp8StyleNet/instance_normalization_2/Reshape/ReadVariableOp2x
:StyleNet/instance_normalization_2/Reshape_1/ReadVariableOp:StyleNet/instance_normalization_2/Reshape_1/ReadVariableOp2t
8StyleNet/instance_normalization_3/Reshape/ReadVariableOp8StyleNet/instance_normalization_3/Reshape/ReadVariableOp2x
:StyleNet/instance_normalization_3/Reshape_1/ReadVariableOp:StyleNet/instance_normalization_3/Reshape_1/ReadVariableOp2t
8StyleNet/instance_normalization_4/Reshape/ReadVariableOp8StyleNet/instance_normalization_4/Reshape/ReadVariableOp2x
:StyleNet/instance_normalization_4/Reshape_1/ReadVariableOp:StyleNet/instance_normalization_4/Reshape_1/ReadVariableOp2t
8StyleNet/instance_normalization_5/Reshape/ReadVariableOp8StyleNet/instance_normalization_5/Reshape/ReadVariableOp2x
:StyleNet/instance_normalization_5/Reshape_1/ReadVariableOp:StyleNet/instance_normalization_5/Reshape_1/ReadVariableOp2r
7StyleNet/residual_block/conv2d_3/BiasAdd/ReadVariableOp7StyleNet/residual_block/conv2d_3/BiasAdd/ReadVariableOp2p
6StyleNet/residual_block/conv2d_3/Conv2D/ReadVariableOp6StyleNet/residual_block/conv2d_3/Conv2D/ReadVariableOp2r
7StyleNet/residual_block/conv2d_4/BiasAdd/ReadVariableOp7StyleNet/residual_block/conv2d_4/BiasAdd/ReadVariableOp2p
6StyleNet/residual_block/conv2d_4/Conv2D/ReadVariableOp6StyleNet/residual_block/conv2d_4/Conv2D/ReadVariableOp2v
9StyleNet/residual_block_1/conv2d_5/BiasAdd/ReadVariableOp9StyleNet/residual_block_1/conv2d_5/BiasAdd/ReadVariableOp2t
8StyleNet/residual_block_1/conv2d_5/Conv2D/ReadVariableOp8StyleNet/residual_block_1/conv2d_5/Conv2D/ReadVariableOp2v
9StyleNet/residual_block_1/conv2d_6/BiasAdd/ReadVariableOp9StyleNet/residual_block_1/conv2d_6/BiasAdd/ReadVariableOp2t
8StyleNet/residual_block_1/conv2d_6/Conv2D/ReadVariableOp8StyleNet/residual_block_1/conv2d_6/Conv2D/ReadVariableOp2v
9StyleNet/residual_block_2/conv2d_7/BiasAdd/ReadVariableOp9StyleNet/residual_block_2/conv2d_7/BiasAdd/ReadVariableOp2t
8StyleNet/residual_block_2/conv2d_7/Conv2D/ReadVariableOp8StyleNet/residual_block_2/conv2d_7/Conv2D/ReadVariableOp2v
9StyleNet/residual_block_2/conv2d_8/BiasAdd/ReadVariableOp9StyleNet/residual_block_2/conv2d_8/BiasAdd/ReadVariableOp2t
8StyleNet/residual_block_2/conv2d_8/Conv2D/ReadVariableOp8StyleNet/residual_block_2/conv2d_8/Conv2D/ReadVariableOp2x
:StyleNet/residual_block_3/conv2d_10/BiasAdd/ReadVariableOp:StyleNet/residual_block_3/conv2d_10/BiasAdd/ReadVariableOp2v
9StyleNet/residual_block_3/conv2d_10/Conv2D/ReadVariableOp9StyleNet/residual_block_3/conv2d_10/Conv2D/ReadVariableOp2v
9StyleNet/residual_block_3/conv2d_9/BiasAdd/ReadVariableOp9StyleNet/residual_block_3/conv2d_9/BiasAdd/ReadVariableOp2t
8StyleNet/residual_block_3/conv2d_9/Conv2D/ReadVariableOp8StyleNet/residual_block_3/conv2d_9/Conv2D/ReadVariableOp2x
:StyleNet/residual_block_4/conv2d_11/BiasAdd/ReadVariableOp:StyleNet/residual_block_4/conv2d_11/BiasAdd/ReadVariableOp2v
9StyleNet/residual_block_4/conv2d_11/Conv2D/ReadVariableOp9StyleNet/residual_block_4/conv2d_11/Conv2D/ReadVariableOp2x
:StyleNet/residual_block_4/conv2d_12/BiasAdd/ReadVariableOp:StyleNet/residual_block_4/conv2d_12/BiasAdd/ReadVariableOp2v
9StyleNet/residual_block_4/conv2d_12/Conv2D/ReadVariableOp9StyleNet/residual_block_4/conv2d_12/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:???????????
&
_user_specified_nameconv2d_input
?
?
-__inference_residual_block_layer_call_fn_4540
input_1!
unknown:00
	unknown_0:0#
	unknown_1:00
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_residual_block_layer_call_and_return_conditional_losses_45262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????Z?0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????Z?0
!
_user_specified_name	input_1
?
?
/__inference_residual_block_4_layer_call_fn_4836
input_1!
unknown:00
	unknown_0:0#
	unknown_1:00
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_4_layer_call_and_return_conditional_losses_48222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????Z?0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????Z?0
!
_user_specified_name	input_1
?0
?
R__inference_instance_normalization_5_layer_call_and_return_conditional_losses_5393

inputs-
reshape_readvariableop_resource:/
!reshape_1_readvariableop_resource:
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:?????????2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
moments/variance?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????2
batchnorm/addx
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/Rsqrt?
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:?????????2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2
batchnorm/mul_1?
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/mul_2?
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?/
?
P__inference_instance_normalization_layer_call_and_return_conditional_losses_7363

inputs-
reshape_readvariableop_resource:/
!reshape_1_readvariableop_resource:
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:?????????2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*1
_output_shapes
:???????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
moments/variance?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????2
batchnorm/addx
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/Rsqrt?
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:?????????2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*1
_output_shapes
:???????????2
batchnorm/mul_1?
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/mul_2?
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*1
_output_shapes
:???????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?0
?
R__inference_instance_normalization_3_layer_call_and_return_conditional_losses_7587

inputs-
reshape_readvariableop_resource: /
!reshape_1_readvariableop_resource: 
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:????????? 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(2
moments/variance?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
: *
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
: 2	
Reshape?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2
	Reshape_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:????????? 2
batchnorm/addx
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:????????? 2
batchnorm/Rsqrt?
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:????????? 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
batchnorm/mul_1?
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:????????? 2
batchnorm/mul_2?
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:????????? 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
E
)__inference_activation_layer_call_fn_7382

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_50212
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
B__inference_conv2d_8_layer_call_and_return_conditional_losses_7857

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
b
F__inference_activation_4_layer_call_and_return_conditional_losses_7663

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?/
?
R__inference_instance_normalization_2_layer_call_and_return_conditional_losses_7525

inputs-
reshape_readvariableop_resource:0/
!reshape_1_readvariableop_resource:0
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:?????????02
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:?????????Z?02
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2
moments/variance?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:0*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         0   2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:02	
Reshape?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:0*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         0   2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:02
	Reshape_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????02
batchnorm/addx
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:?????????02
batchnorm/Rsqrt?
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:?????????02
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*0
_output_shapes
:?????????Z?02
batchnorm/mul_1?
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????02
batchnorm/mul_2?
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????02
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*0
_output_shapes
:?????????Z?02
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
/__inference_residual_block_3_layer_call_fn_4762
input_1!
unknown:00
	unknown_0:0#
	unknown_1:00
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_3_layer_call_and_return_conditional_losses_47482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????Z?0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????Z?0
!
_user_specified_name	input_1
?
?
"__inference_signature_wrapper_6353
conv2d_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7: 0
	unknown_8:0
	unknown_9:0

unknown_10:0$

unknown_11:00

unknown_12:0$

unknown_13:00

unknown_14:0$

unknown_15:00

unknown_16:0$

unknown_17:00

unknown_18:0$

unknown_19:00

unknown_20:0$

unknown_21:00

unknown_22:0$

unknown_23:00

unknown_24:0$

unknown_25:00

unknown_26:0$

unknown_27:00

unknown_28:0$

unknown_29:00

unknown_30:0$

unknown_31: 0

unknown_32: 

unknown_33: 

unknown_34: $

unknown_35: 

unknown_36:

unknown_37:

unknown_38:$

unknown_39:

unknown_40:

unknown_41:

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_44862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:???????????
&
_user_specified_nameconv2d_input
?
?
J__inference_residual_block_2_layer_call_and_return_conditional_losses_4674
input_1'
conv2d_7_4650:00
conv2d_7_4652:0'
conv2d_8_4666:00
conv2d_8_4668:0
identity?? conv2d_7/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_7_4650conv2d_7_4652*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_46492"
 conv2d_7/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0conv2d_8_4666conv2d_8_4668*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_46652"
 conv2d_8/StatefulPartitionedCall?
add/addAddV2input_1)conv2d_8/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:?????????Z?02	
add/addh

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:?????????Z?02

re_lu/Relu?
IdentityIdentityre_lu/Relu:activations:0!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????Z?0: : : : 2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:Y U
0
_output_shapes
:?????????Z?0
!
_user_specified_name	input_1
?
?
B__inference_conv2d_5_layer_call_and_return_conditional_losses_4575

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_1_layer_call_fn_7453

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_instance_normalization_1_layer_call_and_return_conditional_losses_50822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
H__inference_residual_block_layer_call_and_return_conditional_losses_4526
input_1'
conv2d_3_4502:00
conv2d_3_4504:0'
conv2d_4_4518:00
conv2d_4_4520:0
identity?? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_3_4502conv2d_3_4504*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_45012"
 conv2d_3/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_4518conv2d_4_4520*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_45172"
 conv2d_4/StatefulPartitionedCall?
add/addAddV2input_1)conv2d_4/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:?????????Z?02	
add/addh

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:?????????Z?02

re_lu/Relu?
IdentityIdentityre_lu/Relu:activations:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????Z?0: : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:Y U
0
_output_shapes
:?????????Z?0
!
_user_specified_name	input_1
?

?
C__inference_conv2d_12_layer_call_and_return_conditional_losses_7935

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
(__inference_conv2d_13_layer_call_fn_7687

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_53442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_activation_5_layer_call_fn_7749

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_54042
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?^
?
__inference__traced_save_8099
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop;
7savev2_instance_normalization_gamma_read_readvariableop:
6savev2_instance_normalization_beta_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop=
9savev2_instance_normalization_1_gamma_read_readvariableop<
8savev2_instance_normalization_1_beta_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop=
9savev2_instance_normalization_2_gamma_read_readvariableop<
8savev2_instance_normalization_2_beta_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop=
9savev2_instance_normalization_3_gamma_read_readvariableop<
8savev2_instance_normalization_3_beta_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop=
9savev2_instance_normalization_4_gamma_read_readvariableop<
8savev2_instance_normalization_4_beta_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop=
9savev2_instance_normalization_5_gamma_read_readvariableop<
8savev2_instance_normalization_5_beta_read_readvariableop=
9savev2_residual_block_conv2d_3_kernel_read_readvariableop;
7savev2_residual_block_conv2d_3_bias_read_readvariableop=
9savev2_residual_block_conv2d_4_kernel_read_readvariableop;
7savev2_residual_block_conv2d_4_bias_read_readvariableop?
;savev2_residual_block_1_conv2d_5_kernel_read_readvariableop=
9savev2_residual_block_1_conv2d_5_bias_read_readvariableop?
;savev2_residual_block_1_conv2d_6_kernel_read_readvariableop=
9savev2_residual_block_1_conv2d_6_bias_read_readvariableop?
;savev2_residual_block_2_conv2d_7_kernel_read_readvariableop=
9savev2_residual_block_2_conv2d_7_bias_read_readvariableop?
;savev2_residual_block_2_conv2d_8_kernel_read_readvariableop=
9savev2_residual_block_2_conv2d_8_bias_read_readvariableop?
;savev2_residual_block_3_conv2d_9_kernel_read_readvariableop=
9savev2_residual_block_3_conv2d_9_bias_read_readvariableop@
<savev2_residual_block_3_conv2d_10_kernel_read_readvariableop>
:savev2_residual_block_3_conv2d_10_bias_read_readvariableop@
<savev2_residual_block_4_conv2d_11_kernel_read_readvariableop>
:savev2_residual_block_4_conv2d_11_bias_read_readvariableop@
<savev2_residual_block_4_conv2d_12_kernel_read_readvariableop>
:savev2_residual_block_4_conv2d_12_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*?
value?B?-B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/30/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/31/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop7savev2_instance_normalization_gamma_read_readvariableop6savev2_instance_normalization_beta_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop9savev2_instance_normalization_1_gamma_read_readvariableop8savev2_instance_normalization_1_beta_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop9savev2_instance_normalization_2_gamma_read_readvariableop8savev2_instance_normalization_2_beta_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop9savev2_instance_normalization_3_gamma_read_readvariableop8savev2_instance_normalization_3_beta_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop9savev2_instance_normalization_4_gamma_read_readvariableop8savev2_instance_normalization_4_beta_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop9savev2_instance_normalization_5_gamma_read_readvariableop8savev2_instance_normalization_5_beta_read_readvariableop9savev2_residual_block_conv2d_3_kernel_read_readvariableop7savev2_residual_block_conv2d_3_bias_read_readvariableop9savev2_residual_block_conv2d_4_kernel_read_readvariableop7savev2_residual_block_conv2d_4_bias_read_readvariableop;savev2_residual_block_1_conv2d_5_kernel_read_readvariableop9savev2_residual_block_1_conv2d_5_bias_read_readvariableop;savev2_residual_block_1_conv2d_6_kernel_read_readvariableop9savev2_residual_block_1_conv2d_6_bias_read_readvariableop;savev2_residual_block_2_conv2d_7_kernel_read_readvariableop9savev2_residual_block_2_conv2d_7_bias_read_readvariableop;savev2_residual_block_2_conv2d_8_kernel_read_readvariableop9savev2_residual_block_2_conv2d_8_bias_read_readvariableop;savev2_residual_block_3_conv2d_9_kernel_read_readvariableop9savev2_residual_block_3_conv2d_9_bias_read_readvariableop<savev2_residual_block_3_conv2d_10_kernel_read_readvariableop:savev2_residual_block_3_conv2d_10_bias_read_readvariableop<savev2_residual_block_4_conv2d_11_kernel_read_readvariableop:savev2_residual_block_4_conv2d_11_bias_read_readvariableop<savev2_residual_block_4_conv2d_12_kernel_read_readvariableop:savev2_residual_block_4_conv2d_12_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *;
dtypes1
/2-2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::: : : : : 0:0:0:0: 0: : : : ::::::::00:0:00:0:00:0:00:0:00:0:00:0:00:0:00:0:00:0:00:0: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,	(
&
_output_shapes
: 0: 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
: 0: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:00: 

_output_shapes
:0:,(
&
_output_shapes
:00: 

_output_shapes
:0:,(
&
_output_shapes
:00: 

_output_shapes
:0:,(
&
_output_shapes
:00:  

_output_shapes
:0:,!(
&
_output_shapes
:00: "

_output_shapes
:0:,#(
&
_output_shapes
:00: $

_output_shapes
:0:,%(
&
_output_shapes
:00: &

_output_shapes
:0:,'(
&
_output_shapes
:00: (

_output_shapes
:0:,)(
&
_output_shapes
:00: *

_output_shapes
:0:,+(
&
_output_shapes
:00: ,

_output_shapes
:0:-

_output_shapes
: 
?
G
+__inference_activation_3_layer_call_fn_7606

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_52712
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
'__inference_StyleNet_layer_call_fn_6028
conv2d_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7: 0
	unknown_8:0
	unknown_9:0

unknown_10:0$

unknown_11:00

unknown_12:0$

unknown_13:00

unknown_14:0$

unknown_15:00

unknown_16:0$

unknown_17:00

unknown_18:0$

unknown_19:00

unknown_20:0$

unknown_21:00

unknown_22:0$

unknown_23:00

unknown_24:0$

unknown_25:00

unknown_26:0$

unknown_27:00

unknown_28:0$

unknown_29:00

unknown_30:0$

unknown_31: 0

unknown_32: 

unknown_33: 

unknown_34: $

unknown_35: 

unknown_36:

unknown_37:

unknown_38:$

unknown_39:

unknown_40:

unknown_41:

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_StyleNet_layer_call_and_return_conditional_losses_58442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:???????????
&
_user_specified_nameconv2d_input
?
?
B__inference_conv2d_7_layer_call_and_return_conditional_losses_4649

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
'__inference_conv2d_9_layer_call_fn_7886

inputs!
unknown:00
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_9_layer_call_and_return_conditional_losses_47232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
/__inference_conv2d_transpose_layer_call_fn_4900

inputs!
unknown: 0
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_48902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
'__inference_conv2d_5_layer_call_fn_7808

inputs!
unknown:00
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_45752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
ݖ
?*
B__inference_StyleNet_layer_call_and_return_conditional_losses_7115

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:D
6instance_normalization_reshape_readvariableop_resource:F
8instance_normalization_reshape_1_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: F
8instance_normalization_1_reshape_readvariableop_resource: H
:instance_normalization_1_reshape_1_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: 06
(conv2d_2_biasadd_readvariableop_resource:0F
8instance_normalization_2_reshape_readvariableop_resource:0H
:instance_normalization_2_reshape_1_readvariableop_resource:0P
6residual_block_conv2d_3_conv2d_readvariableop_resource:00E
7residual_block_conv2d_3_biasadd_readvariableop_resource:0P
6residual_block_conv2d_4_conv2d_readvariableop_resource:00E
7residual_block_conv2d_4_biasadd_readvariableop_resource:0R
8residual_block_1_conv2d_5_conv2d_readvariableop_resource:00G
9residual_block_1_conv2d_5_biasadd_readvariableop_resource:0R
8residual_block_1_conv2d_6_conv2d_readvariableop_resource:00G
9residual_block_1_conv2d_6_biasadd_readvariableop_resource:0R
8residual_block_2_conv2d_7_conv2d_readvariableop_resource:00G
9residual_block_2_conv2d_7_biasadd_readvariableop_resource:0R
8residual_block_2_conv2d_8_conv2d_readvariableop_resource:00G
9residual_block_2_conv2d_8_biasadd_readvariableop_resource:0R
8residual_block_3_conv2d_9_conv2d_readvariableop_resource:00G
9residual_block_3_conv2d_9_biasadd_readvariableop_resource:0S
9residual_block_3_conv2d_10_conv2d_readvariableop_resource:00H
:residual_block_3_conv2d_10_biasadd_readvariableop_resource:0S
9residual_block_4_conv2d_11_conv2d_readvariableop_resource:00H
:residual_block_4_conv2d_11_biasadd_readvariableop_resource:0S
9residual_block_4_conv2d_12_conv2d_readvariableop_resource:00H
:residual_block_4_conv2d_12_biasadd_readvariableop_resource:0S
9conv2d_transpose_conv2d_transpose_readvariableop_resource: 0>
0conv2d_transpose_biasadd_readvariableop_resource: F
8instance_normalization_3_reshape_readvariableop_resource: H
:instance_normalization_3_reshape_1_readvariableop_resource: U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_1_biasadd_readvariableop_resource:F
8instance_normalization_4_reshape_readvariableop_resource:H
:instance_normalization_4_reshape_1_readvariableop_resource:B
(conv2d_13_conv2d_readvariableop_resource:7
)conv2d_13_biasadd_readvariableop_resource:F
8instance_normalization_5_reshape_readvariableop_resource:H
:instance_normalization_5_reshape_1_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?-instance_normalization/Reshape/ReadVariableOp?/instance_normalization/Reshape_1/ReadVariableOp?/instance_normalization_1/Reshape/ReadVariableOp?1instance_normalization_1/Reshape_1/ReadVariableOp?/instance_normalization_2/Reshape/ReadVariableOp?1instance_normalization_2/Reshape_1/ReadVariableOp?/instance_normalization_3/Reshape/ReadVariableOp?1instance_normalization_3/Reshape_1/ReadVariableOp?/instance_normalization_4/Reshape/ReadVariableOp?1instance_normalization_4/Reshape_1/ReadVariableOp?/instance_normalization_5/Reshape/ReadVariableOp?1instance_normalization_5/Reshape_1/ReadVariableOp?.residual_block/conv2d_3/BiasAdd/ReadVariableOp?-residual_block/conv2d_3/Conv2D/ReadVariableOp?.residual_block/conv2d_4/BiasAdd/ReadVariableOp?-residual_block/conv2d_4/Conv2D/ReadVariableOp?0residual_block_1/conv2d_5/BiasAdd/ReadVariableOp?/residual_block_1/conv2d_5/Conv2D/ReadVariableOp?0residual_block_1/conv2d_6/BiasAdd/ReadVariableOp?/residual_block_1/conv2d_6/Conv2D/ReadVariableOp?0residual_block_2/conv2d_7/BiasAdd/ReadVariableOp?/residual_block_2/conv2d_7/Conv2D/ReadVariableOp?0residual_block_2/conv2d_8/BiasAdd/ReadVariableOp?/residual_block_2/conv2d_8/Conv2D/ReadVariableOp?1residual_block_3/conv2d_10/BiasAdd/ReadVariableOp?0residual_block_3/conv2d_10/Conv2D/ReadVariableOp?0residual_block_3/conv2d_9/BiasAdd/ReadVariableOp?/residual_block_3/conv2d_9/Conv2D/ReadVariableOp?1residual_block_4/conv2d_11/BiasAdd/ReadVariableOp?0residual_block_4/conv2d_11/Conv2D/ReadVariableOp?1residual_block_4/conv2d_12/BiasAdd/ReadVariableOp?0residual_block_4/conv2d_12/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
instance_normalization/ShapeShapeconv2d/BiasAdd:output:0*
T0*
_output_shapes
:2
instance_normalization/Shape?
*instance_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*instance_normalization/strided_slice/stack?
,instance_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,instance_normalization/strided_slice/stack_1?
,instance_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,instance_normalization/strided_slice/stack_2?
$instance_normalization/strided_sliceStridedSlice%instance_normalization/Shape:output:03instance_normalization/strided_slice/stack:output:05instance_normalization/strided_slice/stack_1:output:05instance_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$instance_normalization/strided_slice?
,instance_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,instance_normalization/strided_slice_1/stack?
.instance_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization/strided_slice_1/stack_1?
.instance_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization/strided_slice_1/stack_2?
&instance_normalization/strided_slice_1StridedSlice%instance_normalization/Shape:output:05instance_normalization/strided_slice_1/stack:output:07instance_normalization/strided_slice_1/stack_1:output:07instance_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&instance_normalization/strided_slice_1?
,instance_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,instance_normalization/strided_slice_2/stack?
.instance_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization/strided_slice_2/stack_1?
.instance_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization/strided_slice_2/stack_2?
&instance_normalization/strided_slice_2StridedSlice%instance_normalization/Shape:output:05instance_normalization/strided_slice_2/stack:output:07instance_normalization/strided_slice_2/stack_1:output:07instance_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&instance_normalization/strided_slice_2?
,instance_normalization/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,instance_normalization/strided_slice_3/stack?
.instance_normalization/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization/strided_slice_3/stack_1?
.instance_normalization/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization/strided_slice_3/stack_2?
&instance_normalization/strided_slice_3StridedSlice%instance_normalization/Shape:output:05instance_normalization/strided_slice_3/stack:output:07instance_normalization/strided_slice_3/stack_1:output:07instance_normalization/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&instance_normalization/strided_slice_3?
5instance_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      27
5instance_normalization/moments/mean/reduction_indices?
#instance_normalization/moments/meanMeanconv2d/BiasAdd:output:0>instance_normalization/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2%
#instance_normalization/moments/mean?
+instance_normalization/moments/StopGradientStopGradient,instance_normalization/moments/mean:output:0*
T0*/
_output_shapes
:?????????2-
+instance_normalization/moments/StopGradient?
0instance_normalization/moments/SquaredDifferenceSquaredDifferenceconv2d/BiasAdd:output:04instance_normalization/moments/StopGradient:output:0*
T0*1
_output_shapes
:???????????22
0instance_normalization/moments/SquaredDifference?
9instance_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2;
9instance_normalization/moments/variance/reduction_indices?
'instance_normalization/moments/varianceMean4instance_normalization/moments/SquaredDifference:z:0Binstance_normalization/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2)
'instance_normalization/moments/variance?
-instance_normalization/Reshape/ReadVariableOpReadVariableOp6instance_normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02/
-instance_normalization/Reshape/ReadVariableOp?
$instance_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2&
$instance_normalization/Reshape/shape?
instance_normalization/ReshapeReshape5instance_normalization/Reshape/ReadVariableOp:value:0-instance_normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2 
instance_normalization/Reshape?
/instance_normalization/Reshape_1/ReadVariableOpReadVariableOp8instance_normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype021
/instance_normalization/Reshape_1/ReadVariableOp?
&instance_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2(
&instance_normalization/Reshape_1/shape?
 instance_normalization/Reshape_1Reshape7instance_normalization/Reshape_1/ReadVariableOp:value:0/instance_normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2"
 instance_normalization/Reshape_1?
&instance_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&instance_normalization/batchnorm/add/y?
$instance_normalization/batchnorm/addAddV20instance_normalization/moments/variance:output:0/instance_normalization/batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????2&
$instance_normalization/batchnorm/add?
&instance_normalization/batchnorm/RsqrtRsqrt(instance_normalization/batchnorm/add:z:0*
T0*/
_output_shapes
:?????????2(
&instance_normalization/batchnorm/Rsqrt?
$instance_normalization/batchnorm/mulMul*instance_normalization/batchnorm/Rsqrt:y:0'instance_normalization/Reshape:output:0*
T0*/
_output_shapes
:?????????2&
$instance_normalization/batchnorm/mul?
&instance_normalization/batchnorm/mul_1Mulconv2d/BiasAdd:output:0(instance_normalization/batchnorm/mul:z:0*
T0*1
_output_shapes
:???????????2(
&instance_normalization/batchnorm/mul_1?
&instance_normalization/batchnorm/mul_2Mul,instance_normalization/moments/mean:output:0(instance_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????2(
&instance_normalization/batchnorm/mul_2?
$instance_normalization/batchnorm/subSub)instance_normalization/Reshape_1:output:0*instance_normalization/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????2&
$instance_normalization/batchnorm/sub?
&instance_normalization/batchnorm/add_1AddV2*instance_normalization/batchnorm/mul_1:z:0(instance_normalization/batchnorm/sub:z:0*
T0*1
_output_shapes
:???????????2(
&instance_normalization/batchnorm/add_1?
activation/ReluRelu*instance_normalization/batchnorm/add_1:z:0*
T0*1
_output_shapes
:???????????2
activation/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dactivation/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_1/BiasAdd?
instance_normalization_1/ShapeShapeconv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2 
instance_normalization_1/Shape?
,instance_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,instance_normalization_1/strided_slice/stack?
.instance_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_1/strided_slice/stack_1?
.instance_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_1/strided_slice/stack_2?
&instance_normalization_1/strided_sliceStridedSlice'instance_normalization_1/Shape:output:05instance_normalization_1/strided_slice/stack:output:07instance_normalization_1/strided_slice/stack_1:output:07instance_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&instance_normalization_1/strided_slice?
.instance_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_1/strided_slice_1/stack?
0instance_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_1/strided_slice_1/stack_1?
0instance_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_1/strided_slice_1/stack_2?
(instance_normalization_1/strided_slice_1StridedSlice'instance_normalization_1/Shape:output:07instance_normalization_1/strided_slice_1/stack:output:09instance_normalization_1/strided_slice_1/stack_1:output:09instance_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_1/strided_slice_1?
.instance_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_1/strided_slice_2/stack?
0instance_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_1/strided_slice_2/stack_1?
0instance_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_1/strided_slice_2/stack_2?
(instance_normalization_1/strided_slice_2StridedSlice'instance_normalization_1/Shape:output:07instance_normalization_1/strided_slice_2/stack:output:09instance_normalization_1/strided_slice_2/stack_1:output:09instance_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_1/strided_slice_2?
.instance_normalization_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_1/strided_slice_3/stack?
0instance_normalization_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_1/strided_slice_3/stack_1?
0instance_normalization_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_1/strided_slice_3/stack_2?
(instance_normalization_1/strided_slice_3StridedSlice'instance_normalization_1/Shape:output:07instance_normalization_1/strided_slice_3/stack:output:09instance_normalization_1/strided_slice_3/stack_1:output:09instance_normalization_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_1/strided_slice_3?
7instance_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7instance_normalization_1/moments/mean/reduction_indices?
%instance_normalization_1/moments/meanMeanconv2d_1/BiasAdd:output:0@instance_normalization_1/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(2'
%instance_normalization_1/moments/mean?
-instance_normalization_1/moments/StopGradientStopGradient.instance_normalization_1/moments/mean:output:0*
T0*/
_output_shapes
:????????? 2/
-instance_normalization_1/moments/StopGradient?
2instance_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv2d_1/BiasAdd:output:06instance_normalization_1/moments/StopGradient:output:0*
T0*1
_output_shapes
:??????????? 24
2instance_normalization_1/moments/SquaredDifference?
;instance_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2=
;instance_normalization_1/moments/variance/reduction_indices?
)instance_normalization_1/moments/varianceMean6instance_normalization_1/moments/SquaredDifference:z:0Dinstance_normalization_1/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(2+
)instance_normalization_1/moments/variance?
/instance_normalization_1/Reshape/ReadVariableOpReadVariableOp8instance_normalization_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype021
/instance_normalization_1/Reshape/ReadVariableOp?
&instance_normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&instance_normalization_1/Reshape/shape?
 instance_normalization_1/ReshapeReshape7instance_normalization_1/Reshape/ReadVariableOp:value:0/instance_normalization_1/Reshape/shape:output:0*
T0*&
_output_shapes
: 2"
 instance_normalization_1/Reshape?
1instance_normalization_1/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_1_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype023
1instance_normalization_1/Reshape_1/ReadVariableOp?
(instance_normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2*
(instance_normalization_1/Reshape_1/shape?
"instance_normalization_1/Reshape_1Reshape9instance_normalization_1/Reshape_1/ReadVariableOp:value:01instance_normalization_1/Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2$
"instance_normalization_1/Reshape_1?
(instance_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2*
(instance_normalization_1/batchnorm/add/y?
&instance_normalization_1/batchnorm/addAddV22instance_normalization_1/moments/variance:output:01instance_normalization_1/batchnorm/add/y:output:0*
T0*/
_output_shapes
:????????? 2(
&instance_normalization_1/batchnorm/add?
(instance_normalization_1/batchnorm/RsqrtRsqrt*instance_normalization_1/batchnorm/add:z:0*
T0*/
_output_shapes
:????????? 2*
(instance_normalization_1/batchnorm/Rsqrt?
&instance_normalization_1/batchnorm/mulMul,instance_normalization_1/batchnorm/Rsqrt:y:0)instance_normalization_1/Reshape:output:0*
T0*/
_output_shapes
:????????? 2(
&instance_normalization_1/batchnorm/mul?
(instance_normalization_1/batchnorm/mul_1Mulconv2d_1/BiasAdd:output:0*instance_normalization_1/batchnorm/mul:z:0*
T0*1
_output_shapes
:??????????? 2*
(instance_normalization_1/batchnorm/mul_1?
(instance_normalization_1/batchnorm/mul_2Mul.instance_normalization_1/moments/mean:output:0*instance_normalization_1/batchnorm/mul:z:0*
T0*/
_output_shapes
:????????? 2*
(instance_normalization_1/batchnorm/mul_2?
&instance_normalization_1/batchnorm/subSub+instance_normalization_1/Reshape_1:output:0,instance_normalization_1/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:????????? 2(
&instance_normalization_1/batchnorm/sub?
(instance_normalization_1/batchnorm/add_1AddV2,instance_normalization_1/batchnorm/mul_1:z:0*instance_normalization_1/batchnorm/sub:z:0*
T0*1
_output_shapes
:??????????? 2*
(instance_normalization_1/batchnorm/add_1?
activation_1/ReluRelu,instance_normalization_1/batchnorm/add_1:z:0*
T0*1
_output_shapes
:??????????? 2
activation_1/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02
conv2d_2/BiasAdd?
instance_normalization_2/ShapeShapeconv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:2 
instance_normalization_2/Shape?
,instance_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,instance_normalization_2/strided_slice/stack?
.instance_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_2/strided_slice/stack_1?
.instance_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_2/strided_slice/stack_2?
&instance_normalization_2/strided_sliceStridedSlice'instance_normalization_2/Shape:output:05instance_normalization_2/strided_slice/stack:output:07instance_normalization_2/strided_slice/stack_1:output:07instance_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&instance_normalization_2/strided_slice?
.instance_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_2/strided_slice_1/stack?
0instance_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_2/strided_slice_1/stack_1?
0instance_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_2/strided_slice_1/stack_2?
(instance_normalization_2/strided_slice_1StridedSlice'instance_normalization_2/Shape:output:07instance_normalization_2/strided_slice_1/stack:output:09instance_normalization_2/strided_slice_1/stack_1:output:09instance_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_2/strided_slice_1?
.instance_normalization_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_2/strided_slice_2/stack?
0instance_normalization_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_2/strided_slice_2/stack_1?
0instance_normalization_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_2/strided_slice_2/stack_2?
(instance_normalization_2/strided_slice_2StridedSlice'instance_normalization_2/Shape:output:07instance_normalization_2/strided_slice_2/stack:output:09instance_normalization_2/strided_slice_2/stack_1:output:09instance_normalization_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_2/strided_slice_2?
.instance_normalization_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_2/strided_slice_3/stack?
0instance_normalization_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_2/strided_slice_3/stack_1?
0instance_normalization_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_2/strided_slice_3/stack_2?
(instance_normalization_2/strided_slice_3StridedSlice'instance_normalization_2/Shape:output:07instance_normalization_2/strided_slice_3/stack:output:09instance_normalization_2/strided_slice_3/stack_1:output:09instance_normalization_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_2/strided_slice_3?
7instance_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7instance_normalization_2/moments/mean/reduction_indices?
%instance_normalization_2/moments/meanMeanconv2d_2/BiasAdd:output:0@instance_normalization_2/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2'
%instance_normalization_2/moments/mean?
-instance_normalization_2/moments/StopGradientStopGradient.instance_normalization_2/moments/mean:output:0*
T0*/
_output_shapes
:?????????02/
-instance_normalization_2/moments/StopGradient?
2instance_normalization_2/moments/SquaredDifferenceSquaredDifferenceconv2d_2/BiasAdd:output:06instance_normalization_2/moments/StopGradient:output:0*
T0*0
_output_shapes
:?????????Z?024
2instance_normalization_2/moments/SquaredDifference?
;instance_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2=
;instance_normalization_2/moments/variance/reduction_indices?
)instance_normalization_2/moments/varianceMean6instance_normalization_2/moments/SquaredDifference:z:0Dinstance_normalization_2/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2+
)instance_normalization_2/moments/variance?
/instance_normalization_2/Reshape/ReadVariableOpReadVariableOp8instance_normalization_2_reshape_readvariableop_resource*
_output_shapes
:0*
dtype021
/instance_normalization_2/Reshape/ReadVariableOp?
&instance_normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         0   2(
&instance_normalization_2/Reshape/shape?
 instance_normalization_2/ReshapeReshape7instance_normalization_2/Reshape/ReadVariableOp:value:0/instance_normalization_2/Reshape/shape:output:0*
T0*&
_output_shapes
:02"
 instance_normalization_2/Reshape?
1instance_normalization_2/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_2_reshape_1_readvariableop_resource*
_output_shapes
:0*
dtype023
1instance_normalization_2/Reshape_1/ReadVariableOp?
(instance_normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         0   2*
(instance_normalization_2/Reshape_1/shape?
"instance_normalization_2/Reshape_1Reshape9instance_normalization_2/Reshape_1/ReadVariableOp:value:01instance_normalization_2/Reshape_1/shape:output:0*
T0*&
_output_shapes
:02$
"instance_normalization_2/Reshape_1?
(instance_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2*
(instance_normalization_2/batchnorm/add/y?
&instance_normalization_2/batchnorm/addAddV22instance_normalization_2/moments/variance:output:01instance_normalization_2/batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????02(
&instance_normalization_2/batchnorm/add?
(instance_normalization_2/batchnorm/RsqrtRsqrt*instance_normalization_2/batchnorm/add:z:0*
T0*/
_output_shapes
:?????????02*
(instance_normalization_2/batchnorm/Rsqrt?
&instance_normalization_2/batchnorm/mulMul,instance_normalization_2/batchnorm/Rsqrt:y:0)instance_normalization_2/Reshape:output:0*
T0*/
_output_shapes
:?????????02(
&instance_normalization_2/batchnorm/mul?
(instance_normalization_2/batchnorm/mul_1Mulconv2d_2/BiasAdd:output:0*instance_normalization_2/batchnorm/mul:z:0*
T0*0
_output_shapes
:?????????Z?02*
(instance_normalization_2/batchnorm/mul_1?
(instance_normalization_2/batchnorm/mul_2Mul.instance_normalization_2/moments/mean:output:0*instance_normalization_2/batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????02*
(instance_normalization_2/batchnorm/mul_2?
&instance_normalization_2/batchnorm/subSub+instance_normalization_2/Reshape_1:output:0,instance_normalization_2/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????02(
&instance_normalization_2/batchnorm/sub?
(instance_normalization_2/batchnorm/add_1AddV2,instance_normalization_2/batchnorm/mul_1:z:0*instance_normalization_2/batchnorm/sub:z:0*
T0*0
_output_shapes
:?????????Z?02*
(instance_normalization_2/batchnorm/add_1?
activation_2/ReluRelu,instance_normalization_2/batchnorm/add_1:z:0*
T0*0
_output_shapes
:?????????Z?02
activation_2/Relu?
-residual_block/conv2d_3/Conv2D/ReadVariableOpReadVariableOp6residual_block_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02/
-residual_block/conv2d_3/Conv2D/ReadVariableOp?
residual_block/conv2d_3/Conv2DConv2Dactivation_2/Relu:activations:05residual_block/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2 
residual_block/conv2d_3/Conv2D?
.residual_block/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp7residual_block_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype020
.residual_block/conv2d_3/BiasAdd/ReadVariableOp?
residual_block/conv2d_3/BiasAddBiasAdd'residual_block/conv2d_3/Conv2D:output:06residual_block/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02!
residual_block/conv2d_3/BiasAdd?
residual_block/conv2d_3/ReluRelu(residual_block/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
residual_block/conv2d_3/Relu?
-residual_block/conv2d_4/Conv2D/ReadVariableOpReadVariableOp6residual_block_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02/
-residual_block/conv2d_4/Conv2D/ReadVariableOp?
residual_block/conv2d_4/Conv2DConv2D*residual_block/conv2d_3/Relu:activations:05residual_block/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2 
residual_block/conv2d_4/Conv2D?
.residual_block/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp7residual_block_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype020
.residual_block/conv2d_4/BiasAdd/ReadVariableOp?
residual_block/conv2d_4/BiasAddBiasAdd'residual_block/conv2d_4/Conv2D:output:06residual_block/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02!
residual_block/conv2d_4/BiasAdd?
residual_block/add/addAddV2activation_2/Relu:activations:0(residual_block/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
residual_block/add/add?
residual_block/re_lu/ReluReluresidual_block/add/add:z:0*
T0*0
_output_shapes
:?????????Z?02
residual_block/re_lu/Relu?
/residual_block_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp8residual_block_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype021
/residual_block_1/conv2d_5/Conv2D/ReadVariableOp?
 residual_block_1/conv2d_5/Conv2DConv2D'residual_block/re_lu/Relu:activations:07residual_block_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2"
 residual_block_1/conv2d_5/Conv2D?
0residual_block_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype022
0residual_block_1/conv2d_5/BiasAdd/ReadVariableOp?
!residual_block_1/conv2d_5/BiasAddBiasAdd)residual_block_1/conv2d_5/Conv2D:output:08residual_block_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02#
!residual_block_1/conv2d_5/BiasAdd?
residual_block_1/conv2d_5/ReluRelu*residual_block_1/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02 
residual_block_1/conv2d_5/Relu?
/residual_block_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp8residual_block_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype021
/residual_block_1/conv2d_6/Conv2D/ReadVariableOp?
 residual_block_1/conv2d_6/Conv2DConv2D,residual_block_1/conv2d_5/Relu:activations:07residual_block_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2"
 residual_block_1/conv2d_6/Conv2D?
0residual_block_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype022
0residual_block_1/conv2d_6/BiasAdd/ReadVariableOp?
!residual_block_1/conv2d_6/BiasAddBiasAdd)residual_block_1/conv2d_6/Conv2D:output:08residual_block_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02#
!residual_block_1/conv2d_6/BiasAdd?
residual_block_1/add_1/addAddV2'residual_block/re_lu/Relu:activations:0*residual_block_1/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
residual_block_1/add_1/add?
residual_block_1/re_lu_1/ReluReluresidual_block_1/add_1/add:z:0*
T0*0
_output_shapes
:?????????Z?02
residual_block_1/re_lu_1/Relu?
/residual_block_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp8residual_block_2_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype021
/residual_block_2/conv2d_7/Conv2D/ReadVariableOp?
 residual_block_2/conv2d_7/Conv2DConv2D+residual_block_1/re_lu_1/Relu:activations:07residual_block_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2"
 residual_block_2/conv2d_7/Conv2D?
0residual_block_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype022
0residual_block_2/conv2d_7/BiasAdd/ReadVariableOp?
!residual_block_2/conv2d_7/BiasAddBiasAdd)residual_block_2/conv2d_7/Conv2D:output:08residual_block_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02#
!residual_block_2/conv2d_7/BiasAdd?
residual_block_2/conv2d_7/ReluRelu*residual_block_2/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02 
residual_block_2/conv2d_7/Relu?
/residual_block_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp8residual_block_2_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype021
/residual_block_2/conv2d_8/Conv2D/ReadVariableOp?
 residual_block_2/conv2d_8/Conv2DConv2D,residual_block_2/conv2d_7/Relu:activations:07residual_block_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2"
 residual_block_2/conv2d_8/Conv2D?
0residual_block_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype022
0residual_block_2/conv2d_8/BiasAdd/ReadVariableOp?
!residual_block_2/conv2d_8/BiasAddBiasAdd)residual_block_2/conv2d_8/Conv2D:output:08residual_block_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02#
!residual_block_2/conv2d_8/BiasAdd?
residual_block_2/add_2/addAddV2+residual_block_1/re_lu_1/Relu:activations:0*residual_block_2/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
residual_block_2/add_2/add?
residual_block_2/re_lu_2/ReluReluresidual_block_2/add_2/add:z:0*
T0*0
_output_shapes
:?????????Z?02
residual_block_2/re_lu_2/Relu?
/residual_block_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp8residual_block_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype021
/residual_block_3/conv2d_9/Conv2D/ReadVariableOp?
 residual_block_3/conv2d_9/Conv2DConv2D+residual_block_2/re_lu_2/Relu:activations:07residual_block_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2"
 residual_block_3/conv2d_9/Conv2D?
0residual_block_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype022
0residual_block_3/conv2d_9/BiasAdd/ReadVariableOp?
!residual_block_3/conv2d_9/BiasAddBiasAdd)residual_block_3/conv2d_9/Conv2D:output:08residual_block_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02#
!residual_block_3/conv2d_9/BiasAdd?
residual_block_3/conv2d_9/ReluRelu*residual_block_3/conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02 
residual_block_3/conv2d_9/Relu?
0residual_block_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp9residual_block_3_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype022
0residual_block_3/conv2d_10/Conv2D/ReadVariableOp?
!residual_block_3/conv2d_10/Conv2DConv2D,residual_block_3/conv2d_9/Relu:activations:08residual_block_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2#
!residual_block_3/conv2d_10/Conv2D?
1residual_block_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp:residual_block_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype023
1residual_block_3/conv2d_10/BiasAdd/ReadVariableOp?
"residual_block_3/conv2d_10/BiasAddBiasAdd*residual_block_3/conv2d_10/Conv2D:output:09residual_block_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02$
"residual_block_3/conv2d_10/BiasAdd?
residual_block_3/add_3/addAddV2+residual_block_2/re_lu_2/Relu:activations:0+residual_block_3/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
residual_block_3/add_3/add?
residual_block_3/re_lu_3/ReluReluresidual_block_3/add_3/add:z:0*
T0*0
_output_shapes
:?????????Z?02
residual_block_3/re_lu_3/Relu?
0residual_block_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOp9residual_block_4_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype022
0residual_block_4/conv2d_11/Conv2D/ReadVariableOp?
!residual_block_4/conv2d_11/Conv2DConv2D+residual_block_3/re_lu_3/Relu:activations:08residual_block_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2#
!residual_block_4/conv2d_11/Conv2D?
1residual_block_4/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp:residual_block_4_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype023
1residual_block_4/conv2d_11/BiasAdd/ReadVariableOp?
"residual_block_4/conv2d_11/BiasAddBiasAdd*residual_block_4/conv2d_11/Conv2D:output:09residual_block_4/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02$
"residual_block_4/conv2d_11/BiasAdd?
residual_block_4/conv2d_11/ReluRelu+residual_block_4/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02!
residual_block_4/conv2d_11/Relu?
0residual_block_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp9residual_block_4_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype022
0residual_block_4/conv2d_12/Conv2D/ReadVariableOp?
!residual_block_4/conv2d_12/Conv2DConv2D-residual_block_4/conv2d_11/Relu:activations:08residual_block_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2#
!residual_block_4/conv2d_12/Conv2D?
1residual_block_4/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp:residual_block_4_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype023
1residual_block_4/conv2d_12/BiasAdd/ReadVariableOp?
"residual_block_4/conv2d_12/BiasAddBiasAdd*residual_block_4/conv2d_12/Conv2D:output:09residual_block_4/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02$
"residual_block_4/conv2d_12/BiasAdd?
residual_block_4/add_4/addAddV2+residual_block_3/re_lu_3/Relu:activations:0+residual_block_4/conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
residual_block_4/add_4/add?
residual_block_4/re_lu_4/ReluReluresidual_block_4/add_4/add:z:0*
T0*0
_output_shapes
:?????????Z?02
residual_block_4/re_lu_4/Relu?
conv2d_transpose/ShapeShape+residual_block_4/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicew
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/1w
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0+residual_block_4/re_lu_4/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_transpose/BiasAdd?
instance_normalization_3/ShapeShape!conv2d_transpose/BiasAdd:output:0*
T0*
_output_shapes
:2 
instance_normalization_3/Shape?
,instance_normalization_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,instance_normalization_3/strided_slice/stack?
.instance_normalization_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_3/strided_slice/stack_1?
.instance_normalization_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_3/strided_slice/stack_2?
&instance_normalization_3/strided_sliceStridedSlice'instance_normalization_3/Shape:output:05instance_normalization_3/strided_slice/stack:output:07instance_normalization_3/strided_slice/stack_1:output:07instance_normalization_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&instance_normalization_3/strided_slice?
.instance_normalization_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_3/strided_slice_1/stack?
0instance_normalization_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_3/strided_slice_1/stack_1?
0instance_normalization_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_3/strided_slice_1/stack_2?
(instance_normalization_3/strided_slice_1StridedSlice'instance_normalization_3/Shape:output:07instance_normalization_3/strided_slice_1/stack:output:09instance_normalization_3/strided_slice_1/stack_1:output:09instance_normalization_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_3/strided_slice_1?
.instance_normalization_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_3/strided_slice_2/stack?
0instance_normalization_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_3/strided_slice_2/stack_1?
0instance_normalization_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_3/strided_slice_2/stack_2?
(instance_normalization_3/strided_slice_2StridedSlice'instance_normalization_3/Shape:output:07instance_normalization_3/strided_slice_2/stack:output:09instance_normalization_3/strided_slice_2/stack_1:output:09instance_normalization_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_3/strided_slice_2?
.instance_normalization_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_3/strided_slice_3/stack?
0instance_normalization_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_3/strided_slice_3/stack_1?
0instance_normalization_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_3/strided_slice_3/stack_2?
(instance_normalization_3/strided_slice_3StridedSlice'instance_normalization_3/Shape:output:07instance_normalization_3/strided_slice_3/stack:output:09instance_normalization_3/strided_slice_3/stack_1:output:09instance_normalization_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_3/strided_slice_3?
7instance_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7instance_normalization_3/moments/mean/reduction_indices?
%instance_normalization_3/moments/meanMean!conv2d_transpose/BiasAdd:output:0@instance_normalization_3/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(2'
%instance_normalization_3/moments/mean?
-instance_normalization_3/moments/StopGradientStopGradient.instance_normalization_3/moments/mean:output:0*
T0*/
_output_shapes
:????????? 2/
-instance_normalization_3/moments/StopGradient?
2instance_normalization_3/moments/SquaredDifferenceSquaredDifference!conv2d_transpose/BiasAdd:output:06instance_normalization_3/moments/StopGradient:output:0*
T0*1
_output_shapes
:??????????? 24
2instance_normalization_3/moments/SquaredDifference?
;instance_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2=
;instance_normalization_3/moments/variance/reduction_indices?
)instance_normalization_3/moments/varianceMean6instance_normalization_3/moments/SquaredDifference:z:0Dinstance_normalization_3/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(2+
)instance_normalization_3/moments/variance?
/instance_normalization_3/Reshape/ReadVariableOpReadVariableOp8instance_normalization_3_reshape_readvariableop_resource*
_output_shapes
: *
dtype021
/instance_normalization_3/Reshape/ReadVariableOp?
&instance_normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&instance_normalization_3/Reshape/shape?
 instance_normalization_3/ReshapeReshape7instance_normalization_3/Reshape/ReadVariableOp:value:0/instance_normalization_3/Reshape/shape:output:0*
T0*&
_output_shapes
: 2"
 instance_normalization_3/Reshape?
1instance_normalization_3/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_3_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype023
1instance_normalization_3/Reshape_1/ReadVariableOp?
(instance_normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2*
(instance_normalization_3/Reshape_1/shape?
"instance_normalization_3/Reshape_1Reshape9instance_normalization_3/Reshape_1/ReadVariableOp:value:01instance_normalization_3/Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2$
"instance_normalization_3/Reshape_1?
(instance_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2*
(instance_normalization_3/batchnorm/add/y?
&instance_normalization_3/batchnorm/addAddV22instance_normalization_3/moments/variance:output:01instance_normalization_3/batchnorm/add/y:output:0*
T0*/
_output_shapes
:????????? 2(
&instance_normalization_3/batchnorm/add?
(instance_normalization_3/batchnorm/RsqrtRsqrt*instance_normalization_3/batchnorm/add:z:0*
T0*/
_output_shapes
:????????? 2*
(instance_normalization_3/batchnorm/Rsqrt?
&instance_normalization_3/batchnorm/mulMul,instance_normalization_3/batchnorm/Rsqrt:y:0)instance_normalization_3/Reshape:output:0*
T0*/
_output_shapes
:????????? 2(
&instance_normalization_3/batchnorm/mul?
(instance_normalization_3/batchnorm/mul_1Mul!conv2d_transpose/BiasAdd:output:0*instance_normalization_3/batchnorm/mul:z:0*
T0*1
_output_shapes
:??????????? 2*
(instance_normalization_3/batchnorm/mul_1?
(instance_normalization_3/batchnorm/mul_2Mul.instance_normalization_3/moments/mean:output:0*instance_normalization_3/batchnorm/mul:z:0*
T0*/
_output_shapes
:????????? 2*
(instance_normalization_3/batchnorm/mul_2?
&instance_normalization_3/batchnorm/subSub+instance_normalization_3/Reshape_1:output:0,instance_normalization_3/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:????????? 2(
&instance_normalization_3/batchnorm/sub?
(instance_normalization_3/batchnorm/add_1AddV2,instance_normalization_3/batchnorm/mul_1:z:0*instance_normalization_3/batchnorm/sub:z:0*
T0*1
_output_shapes
:??????????? 2*
(instance_normalization_3/batchnorm/add_1?
activation_3/ReluRelu,instance_normalization_3/batchnorm/add_1:z:0*
T0*1
_output_shapes
:??????????? 2
activation_3/Relu?
conv2d_transpose_1/ShapeShapeactivation_3/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice{
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_1/stack/1{
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0activation_3/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_1/BiasAdd?
instance_normalization_4/ShapeShape#conv2d_transpose_1/BiasAdd:output:0*
T0*
_output_shapes
:2 
instance_normalization_4/Shape?
,instance_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,instance_normalization_4/strided_slice/stack?
.instance_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_4/strided_slice/stack_1?
.instance_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_4/strided_slice/stack_2?
&instance_normalization_4/strided_sliceStridedSlice'instance_normalization_4/Shape:output:05instance_normalization_4/strided_slice/stack:output:07instance_normalization_4/strided_slice/stack_1:output:07instance_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&instance_normalization_4/strided_slice?
.instance_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_4/strided_slice_1/stack?
0instance_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_4/strided_slice_1/stack_1?
0instance_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_4/strided_slice_1/stack_2?
(instance_normalization_4/strided_slice_1StridedSlice'instance_normalization_4/Shape:output:07instance_normalization_4/strided_slice_1/stack:output:09instance_normalization_4/strided_slice_1/stack_1:output:09instance_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_4/strided_slice_1?
.instance_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_4/strided_slice_2/stack?
0instance_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_4/strided_slice_2/stack_1?
0instance_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_4/strided_slice_2/stack_2?
(instance_normalization_4/strided_slice_2StridedSlice'instance_normalization_4/Shape:output:07instance_normalization_4/strided_slice_2/stack:output:09instance_normalization_4/strided_slice_2/stack_1:output:09instance_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_4/strided_slice_2?
.instance_normalization_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_4/strided_slice_3/stack?
0instance_normalization_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_4/strided_slice_3/stack_1?
0instance_normalization_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_4/strided_slice_3/stack_2?
(instance_normalization_4/strided_slice_3StridedSlice'instance_normalization_4/Shape:output:07instance_normalization_4/strided_slice_3/stack:output:09instance_normalization_4/strided_slice_3/stack_1:output:09instance_normalization_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_4/strided_slice_3?
7instance_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7instance_normalization_4/moments/mean/reduction_indices?
%instance_normalization_4/moments/meanMean#conv2d_transpose_1/BiasAdd:output:0@instance_normalization_4/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2'
%instance_normalization_4/moments/mean?
-instance_normalization_4/moments/StopGradientStopGradient.instance_normalization_4/moments/mean:output:0*
T0*/
_output_shapes
:?????????2/
-instance_normalization_4/moments/StopGradient?
2instance_normalization_4/moments/SquaredDifferenceSquaredDifference#conv2d_transpose_1/BiasAdd:output:06instance_normalization_4/moments/StopGradient:output:0*
T0*1
_output_shapes
:???????????24
2instance_normalization_4/moments/SquaredDifference?
;instance_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2=
;instance_normalization_4/moments/variance/reduction_indices?
)instance_normalization_4/moments/varianceMean6instance_normalization_4/moments/SquaredDifference:z:0Dinstance_normalization_4/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2+
)instance_normalization_4/moments/variance?
/instance_normalization_4/Reshape/ReadVariableOpReadVariableOp8instance_normalization_4_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/instance_normalization_4/Reshape/ReadVariableOp?
&instance_normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2(
&instance_normalization_4/Reshape/shape?
 instance_normalization_4/ReshapeReshape7instance_normalization_4/Reshape/ReadVariableOp:value:0/instance_normalization_4/Reshape/shape:output:0*
T0*&
_output_shapes
:2"
 instance_normalization_4/Reshape?
1instance_normalization_4/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_4_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1instance_normalization_4/Reshape_1/ReadVariableOp?
(instance_normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2*
(instance_normalization_4/Reshape_1/shape?
"instance_normalization_4/Reshape_1Reshape9instance_normalization_4/Reshape_1/ReadVariableOp:value:01instance_normalization_4/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2$
"instance_normalization_4/Reshape_1?
(instance_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2*
(instance_normalization_4/batchnorm/add/y?
&instance_normalization_4/batchnorm/addAddV22instance_normalization_4/moments/variance:output:01instance_normalization_4/batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????2(
&instance_normalization_4/batchnorm/add?
(instance_normalization_4/batchnorm/RsqrtRsqrt*instance_normalization_4/batchnorm/add:z:0*
T0*/
_output_shapes
:?????????2*
(instance_normalization_4/batchnorm/Rsqrt?
&instance_normalization_4/batchnorm/mulMul,instance_normalization_4/batchnorm/Rsqrt:y:0)instance_normalization_4/Reshape:output:0*
T0*/
_output_shapes
:?????????2(
&instance_normalization_4/batchnorm/mul?
(instance_normalization_4/batchnorm/mul_1Mul#conv2d_transpose_1/BiasAdd:output:0*instance_normalization_4/batchnorm/mul:z:0*
T0*1
_output_shapes
:???????????2*
(instance_normalization_4/batchnorm/mul_1?
(instance_normalization_4/batchnorm/mul_2Mul.instance_normalization_4/moments/mean:output:0*instance_normalization_4/batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????2*
(instance_normalization_4/batchnorm/mul_2?
&instance_normalization_4/batchnorm/subSub+instance_normalization_4/Reshape_1:output:0,instance_normalization_4/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????2(
&instance_normalization_4/batchnorm/sub?
(instance_normalization_4/batchnorm/add_1AddV2,instance_normalization_4/batchnorm/mul_1:z:0*instance_normalization_4/batchnorm/sub:z:0*
T0*1
_output_shapes
:???????????2*
(instance_normalization_4/batchnorm/add_1?
activation_4/ReluRelu,instance_normalization_4/batchnorm/add_1:z:0*
T0*1
_output_shapes
:???????????2
activation_4/Relu?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2Dactivation_4/Relu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_13/Conv2D?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_13/BiasAdd?
instance_normalization_5/ShapeShapeconv2d_13/BiasAdd:output:0*
T0*
_output_shapes
:2 
instance_normalization_5/Shape?
,instance_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,instance_normalization_5/strided_slice/stack?
.instance_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_5/strided_slice/stack_1?
.instance_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_5/strided_slice/stack_2?
&instance_normalization_5/strided_sliceStridedSlice'instance_normalization_5/Shape:output:05instance_normalization_5/strided_slice/stack:output:07instance_normalization_5/strided_slice/stack_1:output:07instance_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&instance_normalization_5/strided_slice?
.instance_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_5/strided_slice_1/stack?
0instance_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_5/strided_slice_1/stack_1?
0instance_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_5/strided_slice_1/stack_2?
(instance_normalization_5/strided_slice_1StridedSlice'instance_normalization_5/Shape:output:07instance_normalization_5/strided_slice_1/stack:output:09instance_normalization_5/strided_slice_1/stack_1:output:09instance_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_5/strided_slice_1?
.instance_normalization_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_5/strided_slice_2/stack?
0instance_normalization_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_5/strided_slice_2/stack_1?
0instance_normalization_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_5/strided_slice_2/stack_2?
(instance_normalization_5/strided_slice_2StridedSlice'instance_normalization_5/Shape:output:07instance_normalization_5/strided_slice_2/stack:output:09instance_normalization_5/strided_slice_2/stack_1:output:09instance_normalization_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_5/strided_slice_2?
.instance_normalization_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.instance_normalization_5/strided_slice_3/stack?
0instance_normalization_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_5/strided_slice_3/stack_1?
0instance_normalization_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0instance_normalization_5/strided_slice_3/stack_2?
(instance_normalization_5/strided_slice_3StridedSlice'instance_normalization_5/Shape:output:07instance_normalization_5/strided_slice_3/stack:output:09instance_normalization_5/strided_slice_3/stack_1:output:09instance_normalization_5/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(instance_normalization_5/strided_slice_3?
7instance_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7instance_normalization_5/moments/mean/reduction_indices?
%instance_normalization_5/moments/meanMeanconv2d_13/BiasAdd:output:0@instance_normalization_5/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2'
%instance_normalization_5/moments/mean?
-instance_normalization_5/moments/StopGradientStopGradient.instance_normalization_5/moments/mean:output:0*
T0*/
_output_shapes
:?????????2/
-instance_normalization_5/moments/StopGradient?
2instance_normalization_5/moments/SquaredDifferenceSquaredDifferenceconv2d_13/BiasAdd:output:06instance_normalization_5/moments/StopGradient:output:0*
T0*1
_output_shapes
:???????????24
2instance_normalization_5/moments/SquaredDifference?
;instance_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2=
;instance_normalization_5/moments/variance/reduction_indices?
)instance_normalization_5/moments/varianceMean6instance_normalization_5/moments/SquaredDifference:z:0Dinstance_normalization_5/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2+
)instance_normalization_5/moments/variance?
/instance_normalization_5/Reshape/ReadVariableOpReadVariableOp8instance_normalization_5_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/instance_normalization_5/Reshape/ReadVariableOp?
&instance_normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2(
&instance_normalization_5/Reshape/shape?
 instance_normalization_5/ReshapeReshape7instance_normalization_5/Reshape/ReadVariableOp:value:0/instance_normalization_5/Reshape/shape:output:0*
T0*&
_output_shapes
:2"
 instance_normalization_5/Reshape?
1instance_normalization_5/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_5_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1instance_normalization_5/Reshape_1/ReadVariableOp?
(instance_normalization_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2*
(instance_normalization_5/Reshape_1/shape?
"instance_normalization_5/Reshape_1Reshape9instance_normalization_5/Reshape_1/ReadVariableOp:value:01instance_normalization_5/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2$
"instance_normalization_5/Reshape_1?
(instance_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2*
(instance_normalization_5/batchnorm/add/y?
&instance_normalization_5/batchnorm/addAddV22instance_normalization_5/moments/variance:output:01instance_normalization_5/batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????2(
&instance_normalization_5/batchnorm/add?
(instance_normalization_5/batchnorm/RsqrtRsqrt*instance_normalization_5/batchnorm/add:z:0*
T0*/
_output_shapes
:?????????2*
(instance_normalization_5/batchnorm/Rsqrt?
&instance_normalization_5/batchnorm/mulMul,instance_normalization_5/batchnorm/Rsqrt:y:0)instance_normalization_5/Reshape:output:0*
T0*/
_output_shapes
:?????????2(
&instance_normalization_5/batchnorm/mul?
(instance_normalization_5/batchnorm/mul_1Mulconv2d_13/BiasAdd:output:0*instance_normalization_5/batchnorm/mul:z:0*
T0*1
_output_shapes
:???????????2*
(instance_normalization_5/batchnorm/mul_1?
(instance_normalization_5/batchnorm/mul_2Mul.instance_normalization_5/moments/mean:output:0*instance_normalization_5/batchnorm/mul:z:0*
T0*/
_output_shapes
:?????????2*
(instance_normalization_5/batchnorm/mul_2?
&instance_normalization_5/batchnorm/subSub+instance_normalization_5/Reshape_1:output:0,instance_normalization_5/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:?????????2(
&instance_normalization_5/batchnorm/sub?
(instance_normalization_5/batchnorm/add_1AddV2,instance_normalization_5/batchnorm/mul_1:z:0*instance_normalization_5/batchnorm/sub:z:0*
T0*1
_output_shapes
:???????????2*
(instance_normalization_5/batchnorm/add_1?
activation_5/TanhTanh,instance_normalization_5/batchnorm/add_1:z:0*
T0*1
_output_shapes
:???????????2
activation_5/Tanh?
IdentityIdentityactivation_5/Tanh:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp.^instance_normalization/Reshape/ReadVariableOp0^instance_normalization/Reshape_1/ReadVariableOp0^instance_normalization_1/Reshape/ReadVariableOp2^instance_normalization_1/Reshape_1/ReadVariableOp0^instance_normalization_2/Reshape/ReadVariableOp2^instance_normalization_2/Reshape_1/ReadVariableOp0^instance_normalization_3/Reshape/ReadVariableOp2^instance_normalization_3/Reshape_1/ReadVariableOp0^instance_normalization_4/Reshape/ReadVariableOp2^instance_normalization_4/Reshape_1/ReadVariableOp0^instance_normalization_5/Reshape/ReadVariableOp2^instance_normalization_5/Reshape_1/ReadVariableOp/^residual_block/conv2d_3/BiasAdd/ReadVariableOp.^residual_block/conv2d_3/Conv2D/ReadVariableOp/^residual_block/conv2d_4/BiasAdd/ReadVariableOp.^residual_block/conv2d_4/Conv2D/ReadVariableOp1^residual_block_1/conv2d_5/BiasAdd/ReadVariableOp0^residual_block_1/conv2d_5/Conv2D/ReadVariableOp1^residual_block_1/conv2d_6/BiasAdd/ReadVariableOp0^residual_block_1/conv2d_6/Conv2D/ReadVariableOp1^residual_block_2/conv2d_7/BiasAdd/ReadVariableOp0^residual_block_2/conv2d_7/Conv2D/ReadVariableOp1^residual_block_2/conv2d_8/BiasAdd/ReadVariableOp0^residual_block_2/conv2d_8/Conv2D/ReadVariableOp2^residual_block_3/conv2d_10/BiasAdd/ReadVariableOp1^residual_block_3/conv2d_10/Conv2D/ReadVariableOp1^residual_block_3/conv2d_9/BiasAdd/ReadVariableOp0^residual_block_3/conv2d_9/Conv2D/ReadVariableOp2^residual_block_4/conv2d_11/BiasAdd/ReadVariableOp1^residual_block_4/conv2d_11/Conv2D/ReadVariableOp2^residual_block_4/conv2d_12/BiasAdd/ReadVariableOp1^residual_block_4/conv2d_12/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^
-instance_normalization/Reshape/ReadVariableOp-instance_normalization/Reshape/ReadVariableOp2b
/instance_normalization/Reshape_1/ReadVariableOp/instance_normalization/Reshape_1/ReadVariableOp2b
/instance_normalization_1/Reshape/ReadVariableOp/instance_normalization_1/Reshape/ReadVariableOp2f
1instance_normalization_1/Reshape_1/ReadVariableOp1instance_normalization_1/Reshape_1/ReadVariableOp2b
/instance_normalization_2/Reshape/ReadVariableOp/instance_normalization_2/Reshape/ReadVariableOp2f
1instance_normalization_2/Reshape_1/ReadVariableOp1instance_normalization_2/Reshape_1/ReadVariableOp2b
/instance_normalization_3/Reshape/ReadVariableOp/instance_normalization_3/Reshape/ReadVariableOp2f
1instance_normalization_3/Reshape_1/ReadVariableOp1instance_normalization_3/Reshape_1/ReadVariableOp2b
/instance_normalization_4/Reshape/ReadVariableOp/instance_normalization_4/Reshape/ReadVariableOp2f
1instance_normalization_4/Reshape_1/ReadVariableOp1instance_normalization_4/Reshape_1/ReadVariableOp2b
/instance_normalization_5/Reshape/ReadVariableOp/instance_normalization_5/Reshape/ReadVariableOp2f
1instance_normalization_5/Reshape_1/ReadVariableOp1instance_normalization_5/Reshape_1/ReadVariableOp2`
.residual_block/conv2d_3/BiasAdd/ReadVariableOp.residual_block/conv2d_3/BiasAdd/ReadVariableOp2^
-residual_block/conv2d_3/Conv2D/ReadVariableOp-residual_block/conv2d_3/Conv2D/ReadVariableOp2`
.residual_block/conv2d_4/BiasAdd/ReadVariableOp.residual_block/conv2d_4/BiasAdd/ReadVariableOp2^
-residual_block/conv2d_4/Conv2D/ReadVariableOp-residual_block/conv2d_4/Conv2D/ReadVariableOp2d
0residual_block_1/conv2d_5/BiasAdd/ReadVariableOp0residual_block_1/conv2d_5/BiasAdd/ReadVariableOp2b
/residual_block_1/conv2d_5/Conv2D/ReadVariableOp/residual_block_1/conv2d_5/Conv2D/ReadVariableOp2d
0residual_block_1/conv2d_6/BiasAdd/ReadVariableOp0residual_block_1/conv2d_6/BiasAdd/ReadVariableOp2b
/residual_block_1/conv2d_6/Conv2D/ReadVariableOp/residual_block_1/conv2d_6/Conv2D/ReadVariableOp2d
0residual_block_2/conv2d_7/BiasAdd/ReadVariableOp0residual_block_2/conv2d_7/BiasAdd/ReadVariableOp2b
/residual_block_2/conv2d_7/Conv2D/ReadVariableOp/residual_block_2/conv2d_7/Conv2D/ReadVariableOp2d
0residual_block_2/conv2d_8/BiasAdd/ReadVariableOp0residual_block_2/conv2d_8/BiasAdd/ReadVariableOp2b
/residual_block_2/conv2d_8/Conv2D/ReadVariableOp/residual_block_2/conv2d_8/Conv2D/ReadVariableOp2f
1residual_block_3/conv2d_10/BiasAdd/ReadVariableOp1residual_block_3/conv2d_10/BiasAdd/ReadVariableOp2d
0residual_block_3/conv2d_10/Conv2D/ReadVariableOp0residual_block_3/conv2d_10/Conv2D/ReadVariableOp2d
0residual_block_3/conv2d_9/BiasAdd/ReadVariableOp0residual_block_3/conv2d_9/BiasAdd/ReadVariableOp2b
/residual_block_3/conv2d_9/Conv2D/ReadVariableOp/residual_block_3/conv2d_9/Conv2D/ReadVariableOp2f
1residual_block_4/conv2d_11/BiasAdd/ReadVariableOp1residual_block_4/conv2d_11/BiasAdd/ReadVariableOp2d
0residual_block_4/conv2d_11/Conv2D/ReadVariableOp0residual_block_4/conv2d_11/Conv2D/ReadVariableOp2f
1residual_block_4/conv2d_12/BiasAdd/ReadVariableOp1residual_block_4/conv2d_12/BiasAdd/ReadVariableOp2d
0residual_block_4/conv2d_12/Conv2D/ReadVariableOp0residual_block_4/conv2d_12/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
D__inference_activation_layer_call_and_return_conditional_losses_5021

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_10_layer_call_and_return_conditional_losses_7896

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?

?
C__inference_conv2d_12_layer_call_and_return_conditional_losses_4813

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
'__inference_conv2d_4_layer_call_fn_7788

inputs!
unknown:00
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_45172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
'__inference_conv2d_8_layer_call_fn_7866

inputs!
unknown:00
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_46652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?

?
B__inference_conv2d_8_layer_call_and_return_conditional_losses_4665

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
C__inference_conv2d_13_layer_call_and_return_conditional_losses_5344

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?0
?
R__inference_instance_normalization_3_layer_call_and_return_conditional_losses_5260

inputs-
reshape_readvariableop_resource: /
!reshape_1_readvariableop_resource: 
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:????????? 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(2
moments/variance?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
: *
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
: 2	
Reshape?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2
	Reshape_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:????????? 2
batchnorm/addx
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:????????? 2
batchnorm/Rsqrt?
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:????????? 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
batchnorm/mul_1?
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:????????? 2
batchnorm/mul_2?
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:????????? 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
5__inference_instance_normalization_layer_call_fn_7372

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_instance_normalization_layer_call_and_return_conditional_losses_50102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
/__inference_residual_block_2_layer_call_fn_4688
input_1!
unknown:00
	unknown_0:0#
	unknown_1:00
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_residual_block_2_layer_call_and_return_conditional_losses_46742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????Z?0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????Z?0
!
_user_specified_name	input_1
?

?
@__inference_conv2d_layer_call_and_return_conditional_losses_4961

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_activation_5_layer_call_and_return_conditional_losses_5404

inputs
identityh
TanhTanhinputs*
T0*A
_output_shapes/
-:+???????????????????????????2
Tanhv
IdentityIdentityTanh:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_7760

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?02	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????Z?02
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs
?
?
(__inference_conv2d_12_layer_call_fn_7944

inputs!
unknown:00
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_48132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????Z?0: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z?0
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
O
conv2d_input?
serving_default_conv2d_input:0???????????J
activation_5:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
?M
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
layer_with_weights-11
layer-14
layer_with_weights-12
layer-15
layer-16
layer_with_weights-13
layer-17
layer_with_weights-14
layer-18
layer-19
layer_with_weights-15
layer-20
layer_with_weights-16
layer-21
layer-22
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?F
_tf_keras_sequential?F{"name": "StyleNet", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "StyleNet", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 360, 640, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 360, 640, 3]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Addons>InstanceNormalization", "config": {"name": "instance_normalization", "trainable": true, "dtype": "float32", "groups": 16, "axis": -1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Addons>InstanceNormalization", "config": {"name": "instance_normalization_1", "trainable": true, "dtype": "float32", "groups": 32, "axis": -1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Addons>InstanceNormalization", "config": {"name": "instance_normalization_2", "trainable": true, "dtype": "float32", "groups": 48, "axis": -1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "ResidualBlock", "config": {"layer was saved without config": true}}, {"class_name": "ResidualBlock", "config": {"layer was saved without config": true}}, {"class_name": "ResidualBlock", "config": {"layer was saved without config": true}}, {"class_name": "ResidualBlock", "config": {"layer was saved without config": true}}, {"class_name": "ResidualBlock", "config": {"layer was saved without config": true}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Addons>InstanceNormalization", "config": {"name": "instance_normalization_3", "trainable": true, "dtype": "float32", "groups": 32, "axis": -1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Addons>InstanceNormalization", "config": {"name": "instance_normalization_4", "trainable": true, "dtype": "float32", "groups": 16, "axis": -1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Addons>InstanceNormalization", "config": {"name": "instance_normalization_5", "trainable": true, "dtype": "float32", "groups": 3, "axis": -1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "tanh"}}]}, "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360, 640, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 360, 640, 3]}, "float32", "conv2d_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential"}}
?

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?
{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 360, 640, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 360, 640, 3]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360, 640, 3]}}
?
	#gamma
$beta
%trainable_variables
&	variables
'regularization_losses
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "instance_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>InstanceNormalization", "config": {"name": "instance_normalization", "trainable": true, "dtype": "float32", "groups": 16, "axis": -1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}, "shared_object_id": 45}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360, 640, 16]}}
?
)trainable_variables
*	variables
+regularization_losses
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 7}
?


-kernel
.bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360, 640, 16]}}
?
	3gamma
4beta
5trainable_variables
6	variables
7regularization_losses
8	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "instance_normalization_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>InstanceNormalization", "config": {"name": "instance_normalization_1", "trainable": true, "dtype": "float32", "groups": 32, "axis": -1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 12}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 180, 320, 32]}}
?
9trainable_variables
:	variables
;regularization_losses
<	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 14}
?


=kernel
>bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 180, 320, 32]}}
?
	Cgamma
Dbeta
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "instance_normalization_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>InstanceNormalization", "config": {"name": "instance_normalization_2", "trainable": true, "dtype": "float32", "groups": 48, "axis": -1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 48}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 48]}}
?
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 21}
?
	Mconv1
	Nconv2
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_model?{"name": "residual_block", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "ResidualBlock", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 90, 160, 48]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "ResidualBlock"}}
?
	Sconv1
	Tconv2
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_model?{"name": "residual_block_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "ResidualBlock", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 90, 160, 48]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "ResidualBlock"}}
?
	Yconv1
	Zconv2
[trainable_variables
\	variables
]regularization_losses
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_model?{"name": "residual_block_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "ResidualBlock", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 90, 160, 48]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "ResidualBlock"}}
?
	_conv1
	`conv2
atrainable_variables
b	variables
cregularization_losses
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_model?{"name": "residual_block_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "ResidualBlock", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 90, 160, 48]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "ResidualBlock"}}
?
	econv1
	fconv2
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_model?{"name": "residual_block_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "ResidualBlock", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 90, 160, 48]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "ResidualBlock"}}
?

kkernel
lbias
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 48}}, "shared_object_id": 50}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 48]}}
?
	qgamma
rbeta
strainable_variables
t	variables
uregularization_losses
v	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "instance_normalization_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>InstanceNormalization", "config": {"name": "instance_normalization_3", "trainable": true, "dtype": "float32", "groups": 32, "axis": -1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 51}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 180, 320, 32]}}
?
wtrainable_variables
x	variables
yregularization_losses
z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 28}
?

{kernel
|bias
}trainable_variables
~	variables
regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 52}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 180, 320, 32]}}
?

?gamma
	?beta
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "instance_normalization_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>InstanceNormalization", "config": {"name": "instance_normalization_4", "trainable": true, "dtype": "float32", "groups": 16, "axis": -1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 33}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}, "shared_object_id": 53}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360, 640, 16]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 35}
?

?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 36}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 37}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 38, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 54}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360, 640, 16]}}
?

?gamma
	?beta
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "instance_normalization_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>InstanceNormalization", "config": {"name": "instance_normalization_5", "trainable": true, "dtype": "float32", "groups": 3, "axis": -1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 40}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 41, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}, "shared_object_id": 55}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360, 640, 3]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "tanh"}, "shared_object_id": 42}
?
0
1
#2
$3
-4
.5
36
47
=8
>9
C10
D11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
k32
l33
q34
r35
{36
|37
?38
?39
?40
?41
?42
?43"
trackable_list_wrapper
?
0
1
#2
$3
-4
.5
36
47
=8
>9
C10
D11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
k32
l33
q34
r35
{36
|37
?38
?39
?40
?41
?42
?43"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
	variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
':%2conv2d/kernel
:2conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
 	variables
!regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2instance_normalization/gamma
):'2instance_normalization/beta
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
%trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
&	variables
'regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
)trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
*	variables
+regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_1/kernel
: 2conv2d_1/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
/trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
0	variables
1regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:* 2instance_normalization_1/gamma
+:) 2instance_normalization_1/beta
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
5trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
6	variables
7regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
9trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
:	variables
;regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 02conv2d_2/kernel
:02conv2d_2/bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
@	variables
Aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*02instance_normalization_2/gamma
+:)02instance_normalization_2/beta
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
Etrainable_variables
?non_trainable_variables
?metrics
?layer_metrics
F	variables
Gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
Itrainable_variables
?non_trainable_variables
?metrics
?layer_metrics
J	variables
Kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 56}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 58, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}, "shared_object_id": 59}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 48]}}
?

?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 60}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 61}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 62, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}, "shared_object_id": 63}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 48]}}
@
?0
?1
?2
?3"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
Otrainable_variables
?non_trainable_variables
?metrics
?layer_metrics
P	variables
Qregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 64}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 65}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 66, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}, "shared_object_id": 67}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 48]}}
?

?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 68}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 69}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 70, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}, "shared_object_id": 71}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 48]}}
@
?0
?1
?2
?3"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
Utrainable_variables
?non_trainable_variables
?metrics
?layer_metrics
V	variables
Wregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 72}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 73}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 74, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}, "shared_object_id": 75}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 48]}}
?

?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 76}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 77}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 78, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}, "shared_object_id": 79}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 48]}}
@
?0
?1
?2
?3"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
[trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
\	variables
]regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 80}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 81}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 82, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}, "shared_object_id": 83}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 48]}}
?

?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 84}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 85}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 86, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}, "shared_object_id": 87}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 48]}}
@
?0
?1
?2
?3"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
atrainable_variables
?non_trainable_variables
?metrics
?layer_metrics
b	variables
cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 88}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 89}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 90, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}, "shared_object_id": 91}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 48]}}
?

?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 92}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 93}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 94, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}, "shared_object_id": 95}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 48]}}
@
?0
?1
?2
?3"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
gtrainable_variables
?non_trainable_variables
?metrics
?layer_metrics
h	variables
iregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/ 02conv2d_transpose/kernel
#:! 2conv2d_transpose/bias
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
mtrainable_variables
?non_trainable_variables
?metrics
?layer_metrics
n	variables
oregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:* 2instance_normalization_3/gamma
+:) 2instance_normalization_3/beta
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
strainable_variables
?non_trainable_variables
?metrics
?layer_metrics
t	variables
uregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
wtrainable_variables
?non_trainable_variables
?metrics
?layer_metrics
x	variables
yregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
3:1 2conv2d_transpose_1/kernel
%:#2conv2d_transpose_1/bias
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
}trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
~	variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*2instance_normalization_4/gamma
+:)2instance_normalization_4/beta
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_13/kernel
:2conv2d_13/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*2instance_normalization_5/gamma
+:)2instance_normalization_5/beta
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
8:6002residual_block/conv2d_3/kernel
*:(02residual_block/conv2d_3/bias
8:6002residual_block/conv2d_4/kernel
*:(02residual_block/conv2d_4/bias
::8002 residual_block_1/conv2d_5/kernel
,:*02residual_block_1/conv2d_5/bias
::8002 residual_block_1/conv2d_6/kernel
,:*02residual_block_1/conv2d_6/bias
::8002 residual_block_2/conv2d_7/kernel
,:*02residual_block_2/conv2d_7/bias
::8002 residual_block_2/conv2d_8/kernel
,:*02residual_block_2/conv2d_8/bias
::8002 residual_block_3/conv2d_9/kernel
,:*02residual_block_3/conv2d_9/bias
;:9002!residual_block_3/conv2d_10/kernel
-:+02residual_block_3/conv2d_10/bias
;:9002!residual_block_4/conv2d_11/kernel
-:+02residual_block_4/conv2d_11/bias
;:9002!residual_block_4/conv2d_12/kernel
-:+02residual_block_4/conv2d_12/bias
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?metrics
?layer_metrics
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
B__inference_StyleNet_layer_call_and_return_conditional_losses_6734
B__inference_StyleNet_layer_call_and_return_conditional_losses_7115
B__inference_StyleNet_layer_call_and_return_conditional_losses_6143
B__inference_StyleNet_layer_call_and_return_conditional_losses_6258?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_StyleNet_layer_call_fn_5498
'__inference_StyleNet_layer_call_fn_7208
'__inference_StyleNet_layer_call_fn_7301
'__inference_StyleNet_layer_call_fn_6028?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_4486?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *5?2
0?-
conv2d_input???????????
?2?
@__inference_conv2d_layer_call_and_return_conditional_losses_7311?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_conv2d_layer_call_fn_7320?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_instance_normalization_layer_call_and_return_conditional_losses_7363?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_instance_normalization_layer_call_fn_7372?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_activation_layer_call_and_return_conditional_losses_7377?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_activation_layer_call_fn_7382?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_7392?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_1_layer_call_fn_7401?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_instance_normalization_1_layer_call_and_return_conditional_losses_7444?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_instance_normalization_1_layer_call_fn_7453?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_activation_1_layer_call_and_return_conditional_losses_7458?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_activation_1_layer_call_fn_7463?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_7473?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_2_layer_call_fn_7482?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_instance_normalization_2_layer_call_and_return_conditional_losses_7525?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_instance_normalization_2_layer_call_fn_7534?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_activation_2_layer_call_and_return_conditional_losses_7539?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_activation_2_layer_call_fn_7544?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_residual_block_layer_call_and_return_conditional_losses_4526?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
input_1?????????Z?0
?2?
-__inference_residual_block_layer_call_fn_4540?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
input_1?????????Z?0
?2?
J__inference_residual_block_1_layer_call_and_return_conditional_losses_4600?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
input_1?????????Z?0
?2?
/__inference_residual_block_1_layer_call_fn_4614?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
input_1?????????Z?0
?2?
J__inference_residual_block_2_layer_call_and_return_conditional_losses_4674?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
input_1?????????Z?0
?2?
/__inference_residual_block_2_layer_call_fn_4688?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
input_1?????????Z?0
?2?
J__inference_residual_block_3_layer_call_and_return_conditional_losses_4748?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
input_1?????????Z?0
?2?
/__inference_residual_block_3_layer_call_fn_4762?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
input_1?????????Z?0
?2?
J__inference_residual_block_4_layer_call_and_return_conditional_losses_4822?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
input_1?????????Z?0
?2?
/__inference_residual_block_4_layer_call_fn_4836?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
input_1?????????Z?0
?2?
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4890?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????0
?2?
/__inference_conv2d_transpose_layer_call_fn_4900?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????0
?2?
R__inference_instance_normalization_3_layer_call_and_return_conditional_losses_7587?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_instance_normalization_3_layer_call_fn_7596?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_activation_3_layer_call_and_return_conditional_losses_7601?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_activation_3_layer_call_fn_7606?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4934?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
1__inference_conv2d_transpose_1_layer_call_fn_4944?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
R__inference_instance_normalization_4_layer_call_and_return_conditional_losses_7649?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_instance_normalization_4_layer_call_fn_7658?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_activation_4_layer_call_and_return_conditional_losses_7663?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_activation_4_layer_call_fn_7668?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_13_layer_call_and_return_conditional_losses_7678?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_13_layer_call_fn_7687?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_instance_normalization_5_layer_call_and_return_conditional_losses_7730?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_instance_normalization_5_layer_call_fn_7739?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_activation_5_layer_call_and_return_conditional_losses_7744?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_activation_5_layer_call_fn_7749?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_6353conv2d_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_7760?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_3_layer_call_fn_7769?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_7779?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_4_layer_call_fn_7788?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_5_layer_call_and_return_conditional_losses_7799?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_5_layer_call_fn_7808?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_6_layer_call_and_return_conditional_losses_7818?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_6_layer_call_fn_7827?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_7_layer_call_and_return_conditional_losses_7838?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_7_layer_call_fn_7847?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_8_layer_call_and_return_conditional_losses_7857?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_8_layer_call_fn_7866?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_9_layer_call_and_return_conditional_losses_7877?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_9_layer_call_fn_7886?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_10_layer_call_and_return_conditional_losses_7896?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_10_layer_call_fn_7905?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_11_layer_call_and_return_conditional_losses_7916?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_11_layer_call_fn_7925?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_12_layer_call_and_return_conditional_losses_7935?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_12_layer_call_fn_7944?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
B__inference_StyleNet_layer_call_and_return_conditional_losses_6143?F#$-.34=>CD????????????????????klqr{|??????G?D
=?:
0?-
conv2d_input???????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
B__inference_StyleNet_layer_call_and_return_conditional_losses_6258?F#$-.34=>CD????????????????????klqr{|??????G?D
=?:
0?-
conv2d_input???????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
B__inference_StyleNet_layer_call_and_return_conditional_losses_6734?F#$-.34=>CD????????????????????klqr{|??????A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
B__inference_StyleNet_layer_call_and_return_conditional_losses_7115?F#$-.34=>CD????????????????????klqr{|??????A?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
'__inference_StyleNet_layer_call_fn_5498?F#$-.34=>CD????????????????????klqr{|??????G?D
=?:
0?-
conv2d_input???????????
p 

 
? "2?/+????????????????????????????
'__inference_StyleNet_layer_call_fn_6028?F#$-.34=>CD????????????????????klqr{|??????G?D
=?:
0?-
conv2d_input???????????
p

 
? "2?/+????????????????????????????
'__inference_StyleNet_layer_call_fn_7208?F#$-.34=>CD????????????????????klqr{|??????A?>
7?4
*?'
inputs???????????
p 

 
? "2?/+????????????????????????????
'__inference_StyleNet_layer_call_fn_7301?F#$-.34=>CD????????????????????klqr{|??????A?>
7?4
*?'
inputs???????????
p

 
? "2?/+????????????????????????????
__inference__wrapped_model_4486?F#$-.34=>CD????????????????????klqr{|????????<
5?2
0?-
conv2d_input???????????
? "E?B
@
activation_50?-
activation_5????????????
F__inference_activation_1_layer_call_and_return_conditional_losses_7458l9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0??????????? 
? ?
+__inference_activation_1_layer_call_fn_7463_9?6
/?,
*?'
inputs??????????? 
? ""???????????? ?
F__inference_activation_2_layer_call_and_return_conditional_losses_7539j8?5
.?+
)?&
inputs?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
+__inference_activation_2_layer_call_fn_7544]8?5
.?+
)?&
inputs?????????Z?0
? "!??????????Z?0?
F__inference_activation_3_layer_call_and_return_conditional_losses_7601?I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
+__inference_activation_3_layer_call_fn_7606I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
F__inference_activation_4_layer_call_and_return_conditional_losses_7663?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
+__inference_activation_4_layer_call_fn_7668I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
F__inference_activation_5_layer_call_and_return_conditional_losses_7744?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
+__inference_activation_5_layer_call_fn_7749I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
D__inference_activation_layer_call_and_return_conditional_losses_7377l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
)__inference_activation_layer_call_fn_7382_9?6
/?,
*?'
inputs???????????
? ""?????????????
C__inference_conv2d_10_layer_call_and_return_conditional_losses_7896p??8?5
.?+
)?&
inputs?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
(__inference_conv2d_10_layer_call_fn_7905c??8?5
.?+
)?&
inputs?????????Z?0
? "!??????????Z?0?
C__inference_conv2d_11_layer_call_and_return_conditional_losses_7916p??8?5
.?+
)?&
inputs?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
(__inference_conv2d_11_layer_call_fn_7925c??8?5
.?+
)?&
inputs?????????Z?0
? "!??????????Z?0?
C__inference_conv2d_12_layer_call_and_return_conditional_losses_7935p??8?5
.?+
)?&
inputs?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
(__inference_conv2d_12_layer_call_fn_7944c??8?5
.?+
)?&
inputs?????????Z?0
? "!??????????Z?0?
C__inference_conv2d_13_layer_call_and_return_conditional_losses_7678???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
(__inference_conv2d_13_layer_call_fn_7687???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
B__inference_conv2d_1_layer_call_and_return_conditional_losses_7392p-.9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
'__inference_conv2d_1_layer_call_fn_7401c-.9?6
/?,
*?'
inputs???????????
? ""???????????? ?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_7473o=>9?6
/?,
*?'
inputs??????????? 
? ".?+
$?!
0?????????Z?0
? ?
'__inference_conv2d_2_layer_call_fn_7482b=>9?6
/?,
*?'
inputs??????????? 
? "!??????????Z?0?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_7760p??8?5
.?+
)?&
inputs?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
'__inference_conv2d_3_layer_call_fn_7769c??8?5
.?+
)?&
inputs?????????Z?0
? "!??????????Z?0?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_7779p??8?5
.?+
)?&
inputs?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
'__inference_conv2d_4_layer_call_fn_7788c??8?5
.?+
)?&
inputs?????????Z?0
? "!??????????Z?0?
B__inference_conv2d_5_layer_call_and_return_conditional_losses_7799p??8?5
.?+
)?&
inputs?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
'__inference_conv2d_5_layer_call_fn_7808c??8?5
.?+
)?&
inputs?????????Z?0
? "!??????????Z?0?
B__inference_conv2d_6_layer_call_and_return_conditional_losses_7818p??8?5
.?+
)?&
inputs?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
'__inference_conv2d_6_layer_call_fn_7827c??8?5
.?+
)?&
inputs?????????Z?0
? "!??????????Z?0?
B__inference_conv2d_7_layer_call_and_return_conditional_losses_7838p??8?5
.?+
)?&
inputs?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
'__inference_conv2d_7_layer_call_fn_7847c??8?5
.?+
)?&
inputs?????????Z?0
? "!??????????Z?0?
B__inference_conv2d_8_layer_call_and_return_conditional_losses_7857p??8?5
.?+
)?&
inputs?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
'__inference_conv2d_8_layer_call_fn_7866c??8?5
.?+
)?&
inputs?????????Z?0
? "!??????????Z?0?
B__inference_conv2d_9_layer_call_and_return_conditional_losses_7877p??8?5
.?+
)?&
inputs?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
'__inference_conv2d_9_layer_call_fn_7886c??8?5
.?+
)?&
inputs?????????Z?0
? "!??????????Z?0?
@__inference_conv2d_layer_call_and_return_conditional_losses_7311p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
%__inference_conv2d_layer_call_fn_7320c9?6
/?,
*?'
inputs???????????
? ""?????????????
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4934?{|I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
1__inference_conv2d_transpose_1_layer_call_fn_4944?{|I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4890?klI?F
??<
:?7
inputs+???????????????????????????0
? "??<
5?2
0+??????????????????????????? 
? ?
/__inference_conv2d_transpose_layer_call_fn_4900?klI?F
??<
:?7
inputs+???????????????????????????0
? "2?/+??????????????????????????? ?
R__inference_instance_normalization_1_layer_call_and_return_conditional_losses_7444p349?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0??????????? 
? ?
7__inference_instance_normalization_1_layer_call_fn_7453c349?6
/?,
*?'
inputs??????????? 
? ""???????????? ?
R__inference_instance_normalization_2_layer_call_and_return_conditional_losses_7525nCD8?5
.?+
)?&
inputs?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
7__inference_instance_normalization_2_layer_call_fn_7534aCD8?5
.?+
)?&
inputs?????????Z?0
? "!??????????Z?0?
R__inference_instance_normalization_3_layer_call_and_return_conditional_losses_7587?qrI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
7__inference_instance_normalization_3_layer_call_fn_7596?qrI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
R__inference_instance_normalization_4_layer_call_and_return_conditional_losses_7649???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
7__inference_instance_normalization_4_layer_call_fn_7658???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
R__inference_instance_normalization_5_layer_call_and_return_conditional_losses_7730???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
7__inference_instance_normalization_5_layer_call_fn_7739???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
P__inference_instance_normalization_layer_call_and_return_conditional_losses_7363p#$9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
5__inference_instance_normalization_layer_call_fn_7372c#$9?6
/?,
*?'
inputs???????????
? ""?????????????
J__inference_residual_block_1_layer_call_and_return_conditional_losses_4600u????9?6
/?,
*?'
input_1?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
/__inference_residual_block_1_layer_call_fn_4614h????9?6
/?,
*?'
input_1?????????Z?0
? "!??????????Z?0?
J__inference_residual_block_2_layer_call_and_return_conditional_losses_4674u????9?6
/?,
*?'
input_1?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
/__inference_residual_block_2_layer_call_fn_4688h????9?6
/?,
*?'
input_1?????????Z?0
? "!??????????Z?0?
J__inference_residual_block_3_layer_call_and_return_conditional_losses_4748u????9?6
/?,
*?'
input_1?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
/__inference_residual_block_3_layer_call_fn_4762h????9?6
/?,
*?'
input_1?????????Z?0
? "!??????????Z?0?
J__inference_residual_block_4_layer_call_and_return_conditional_losses_4822u????9?6
/?,
*?'
input_1?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
/__inference_residual_block_4_layer_call_fn_4836h????9?6
/?,
*?'
input_1?????????Z?0
? "!??????????Z?0?
H__inference_residual_block_layer_call_and_return_conditional_losses_4526u????9?6
/?,
*?'
input_1?????????Z?0
? ".?+
$?!
0?????????Z?0
? ?
-__inference_residual_block_layer_call_fn_4540h????9?6
/?,
*?'
input_1?????????Z?0
? "!??????????Z?0?
"__inference_signature_wrapper_6353?F#$-.34=>CD????????????????????klqr{|??????O?L
? 
E?B
@
conv2d_input0?-
conv2d_input???????????"E?B
@
activation_50?-
activation_5???????????