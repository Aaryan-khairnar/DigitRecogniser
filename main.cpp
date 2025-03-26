#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>
#include <cmath>
#include <string>
#include <limits>

using namespace std;

//Random no generative functions
struct xorshift{
    unsigned x, y, z, w;

    xorshift():  x(123456789), y(362436069), z(521288629), w(88675123) {}   

    unsigned next(){
        unsigned t = x ^ (x<<11);

        x = y; y = z; z = w;

        return w = w ^ (w>>19) ^ t ^ (t>>8);
    }
}rng;

double rand_01() {
    return rng.next() / 4294967295.0; 
}

//Matrix

class matrix{
    public:    
        int m,n;
        double **a;
    
        //01
        matrix(): n(0), m(0), a(nullptr) {}
        
        //02
        matrix(int rows, int cols){
            n = rows;
            m = cols;
    
            a = new double * [n];
            for(int i=0; i<n; i++){
                a[i] = new double [m];
    
                for (int j=0; j<m; j++){
                    a[i][j]= 0.0;
                }
            }
        }
    
        //03
        matrix(const matrix &x){
            n = x.n;
            m = x.m;
    
            a = new double* [n];
            for(int i=0; i<n; i++){
                a[i] = new double [m];
                
                for(int j=0; j<m; j++){
                    a[i][j] = x.a[i][j];
                }
            }
        }
    
        //destructor
        ~matrix(){
            for(int i=0; i<n; i++){
                delete[] a[i];
            }
            delete[] a;
        }        
    
        //04
        // Swap function for matrix
        void swap(matrix &other) {
            std::swap(n, other.n);
            std::swap(m, other.m);
            std::swap(a, other.a);
        }
        // Assignment operator using copy-and-swap
        matrix& operator=(matrix x) {  // x is taken by value (a copy is made)
            swap(x);  // Swap internals with the copy
            return *this;
        }

        
        //05
        void randomize(){
            for(int i=0; i<n; i++){
                for(int j=0; j<m; j++){
                    a[i][j] = rand_01();
    
                    if(rng.next()%2 == 0) a[i][j] = -a[i][j];
                }
            }
        }
    
        //06
        void zero(){
            for(int i=0; i<n; i++){
                for(int j=0; j<m; j++){
                    a[i][j] = 0.0;
                }
            }
        }
    
        //07 Assume that X has the same dimensions
        void add(matrix x){
            for(int i=0; i<n; i++){
                for(int j=0; j<m; j++){
                    a[i][j] += x[i][j];
                }
            }
        }
    
        double* &operator [](const int &idx){
            return a[idx];
        }
    };
    
    //01 Assume A and B have the same dimensions
    matrix add(matrix a, matrix b){
        matrix ans(a.n, a.m);
        
        for(int i=0; i<a.n; i++){
            for(int j=0; j<a.m; j++){
                ans[i][j] = a[i][j] + b[i][j];
            }
        }
        return ans;
    }
    
    //02 Assume A and B have the same dimensions
    matrix subtract(matrix a, matrix b){
        matrix ans(a.n, a.m);
        
        for(int i=0; i<a.n; i++){
            for(int j=0; j<a.m; j++){
                ans[i][j] = a[i][j] - b[i][j];
            }
        }
        return ans;
    }
    
    //03 Assume A and B have the same dimensions
    matrix term_by_term_mul(matrix a, matrix b){
        matrix ans(a.n, a.m);
        
        for(int i=0; i<a.n; i++){
            for(int j=0; j<a.m; j++){
                ans[i][j] = a[i][j] * b[i][j];
            }
        }
        return ans;
    }
    
    //04
    matrix transpose(matrix a){
        matrix ans(a.m, a.n);  // Swap dimensions

        for(int i = 0; i < a.n; i++){
            for(int j = 0; j < a.m; j++){
                ans[j][i] = a[i][j];
            }
        }
        return ans;
    }
    
    
    
    //05
    matrix multiply(matrix a, matrix b){
        matrix ans(a.n, b.m);
        for(int i = 0; i < a.n; i++){
            for(int j = 0; j < b.m; j++){
                ans[i][j] = 0.0;
                for(int k = 0; k < a.m; k++){
                    ans[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return ans;
    }
    
    
    
    
    //06
    double sigmoid(double x){
        return 1/(1+ exp(-x));
    }
    
    //07
    double sigmoidDer(double x){
        return x * (1 - x);
    }
    
    //08
    matrix sigmoid(matrix a){
        matrix ans(a.n, a.m);
    
        for(int i=0; i<a.n; i++){
            for(int j=0; j<a.m; j++){
                ans[i][j] = sigmoid(a[i][j]);
            }
        }
        return ans;
    }
    
    //09
    matrix sigmoidDer(matrix a){
        matrix ans(a.n, a.m);
        for(int i = 0; i < a.n; i++){
            for(int j = 0; j < a.m; j++){
                ans[i][j] = sigmoidDer(a[i][j]); // Use ans[i][j] instead of ans[j][i]
            }
        }
        return ans;
    }
    


//Neural Network
class NeuralNetwork{
    public:
        int n;
        vector <int> size;
        vector <matrix> w, b, delw, delb; //delta weight and delta bias
        double learning_rate;
    
        NeuralNetwork(vector <int> sz, double alpha){
            n = (int)(sz.size()); //no of layers in neural network
            size = sz;
    
            w.resize(n-1); //n-1 = no of layers of weights/biases
            b.resize(n-1); //a network of 3 layers has weights of 2 layers
            delw.resize(n-1);
            delb.resize(n-1);
    
            for(int i=0; i<n-1; i++){
                w[i]= matrix(size[i], size[i+1]);
                b[i]= matrix(1, size[i+1]);
                delw[i]= matrix(size[i], size[i+1]);
                delb[i]= matrix(1, size[i+1]);
    
                w[i].randomize();
                b[i].randomize();
            }
            
            learning_rate = alpha;
        }
    
        matrix feedForward(matrix input){
            for(int i=0; i< n-1; i++){
                input = sigmoid(add(multiply(input, w[i]), b[i]));
            }
            return input;
        }
    
        void backpropagation(matrix input, matrix Desired_output){ 
            //here "input" = output of our neural network
            vector <matrix> layer_activation;
            matrix delta; //stores the matrices in intermediate operations
            layer_activation.push_back(input); //first layer l00
            
            for(int i=0; i<n-1; i++){
                input = sigmoid(add(multiply(input, w[i]), b[i]));
                layer_activation.push_back(input);
            }
         
            delta = term_by_term_mul(subtract(input, Desired_output), sigmoidDer(layer_activation[n-1]));
    
            delb[n-2].add(delta);
            delw[n-2].add(multiply(transpose(layer_activation[n-2]),delta));
    
            for(int i=n-3; i>=0; i--){
                delta = multiply(delta, transpose(w[i+1]));
                delta = term_by_term_mul(delta, sigmoidDer(layer_activation[i+1]));
                
                delb[i].add(delta);
                delw[i].add(multiply(transpose(layer_activation[i]),delta));
            }
        }    
    
        void train(vector<matrix> input, vector<matrix> output){
    
            for(int i=0; i<n-1; i++){
                delw[i].zero();
                delb[i].zero();
            }
    
            for(int i=0; i<(int)(input.size()); i++){
                backpropagation(input[i], output[i]);
            }
    
            for(int i=0; i<n-1; i++){
    
                for(int j=0; j< delw[i].n; j++){
                    for(int k=0; k< delw[i].m; k++){
                        delw [i][j][k] /= (double) (input.size());
                        w[i][j][k] -= learning_rate * delw[i][j][k];
                    }
                }
    
                for(int j=0; j< delb[i].n; j++){
                    for(int k=0; k< delb[i].m; k++){
                        delb [i][j][k] /= (double) (input.size());
                        b[i][j][k] -= learning_rate * delb[i][j][k];
                    }
                }
            }
        }
    };
    

NeuralNetwork net({784, 16, 16, 10}, 1.1);

// parsing training data
const int BATCH_SIZE = 20;

vector <matrix> train_input, train_output;

vector<int> split(string s){
    int curr = 0;
    vector<int> ans;

    for(int i=0; i<(int)(s.size()); i++){
        if(s[i] ==','){
            ans.push_back(curr);
            curr = 0;
        } else {
            curr *= 10;
            curr += s[i] - '0';
        }
    }

    ans.push_back(curr);
    return ans;
}

void execution_time(){
    cerr<<"Time: "<< (int) (clock() *1000/ CLOCKS_PER_SEC) <<" ms"<<endl;
}

void parse_training_data(){
    ifstream IN ("mnist_train.csv");
    if(!IN.is_open()){
        cerr << "Error: Could not open mnist_train.csv" << endl;
        return;
    }
    string trash;
    vector <int> v;
    matrix input(1, 784), output(1,10);

    train_input.reserve(60000);
    train_output.reserve(60000);

    getline(IN, trash); 
    for(int i=0; i<60000; i++){
        getline(IN, trash);
        v = split(trash);
        if (v.size() != 785) {
            cerr << "Error: Expected 784 elements but got " << v.size() << endl;
            exit(1);
        }
        

        output.zero();
        output[0][v[0]] = 1.0;

        for(int j=1; j<785; j++){
            input[0][j-1] = v[j]/255;
        }

        train_input.push_back(input);
        train_output.push_back(output);
    }

    cerr<<"Training data loaded"<<endl;
    execution_time();
}

void randomshuffle(vector <int> &v){
    for(int i= (int)(v.size())-1; i>=0; i--){
        swap(v[i], v[rng.next() % (i+1)]);
    }
}

// Training
void train(){
    int epoch;
    vector<int> idx;
    vector<matrix> inputs, outputs;
    matrix curr_output;
    double error;

    // Create indices for all training examples
    for(int i = 0; i < 60000; i++){
        idx.push_back(i);
    }

    for(epoch = 1; epoch <= 10; epoch++){
        cerr << "Epoch: " << epoch << " starting." << endl;
        error = 0.0;

        // Shuffle the indices to randomize the order
        randomshuffle(idx);

        // Process batches
        for(int i = 0; i < 60000; i += BATCH_SIZE){
            inputs.clear();
            outputs.clear();

            // Check: ensure we don't go out-of-bounds
            if(i + BATCH_SIZE > 60000){
                break;
            }
    

            for(int j = 0; j < BATCH_SIZE; j++){
                inputs.push_back(train_input[idx[i+j]]);
                outputs.push_back(train_output[idx[i+j]]);
            }
            net.train(inputs, outputs);
        }

        // After training on all batches, compute the error on training set
        for(int i = 0; i < 60000; i++){
            curr_output = net.feedForward(train_input[i]);
            for(int j = 0; j < 10; j++){
                error += (curr_output[0][j] - train_output[i][0][j]) * 
                         (curr_output[0][j] - train_output[i][0][j]);
            }
        }

        error /= 10.0;
        error /= 60000;
        cerr << "Epoch: " << epoch << " finished. Error: " << error << endl;
        execution_time();
        cerr<<endl;
    }
}



// Testing
void test(){
    ifstream IN("mnist_test.csv");
    ofstream OUT("ans.csv");
    if(!IN.is_open()){
        cerr << "Error: Could not open mnist_test.csv" << endl;
        return;
    }
    
    string trash;
    vector<int> v;
    double max_value;
    int index;
    matrix curr_input(1, 784), curr_output;

    OUT << "ImageId,Label" << endl;

    // Skip header
    getline(IN, trash); // read header and discard it
    for(int i = 0; i < 28000; i++){
        if(!getline(IN, trash)){
            cerr << "Reached end of file unexpectedly at line " << i+2 << endl;
            break;
        }
        
        // Debug: print the line read
        // cerr << "Line " << i+2 << ": " << trash << endl;
        
        if(trash.empty()){
            cerr << "Skipping empty line at line " << i+2 << endl;
            i--; // Adjust the count to still read 28000 valid examples
            continue;
        }
        
        v = split(trash);
        if (v.size() != 784) {
            cerr << "Error: Expected 784 elements but got " << v.size() 
                 << " at line " << i+2 << endl;
            exit(1);
        }
        
        for(int j = 0; j < 784; j++){
            curr_input[0][j] = v[j+1] / 255.0; // Pixel data starts at v[1]
        }

        curr_output = net.feedForward(curr_input); // Run network on input

        max_value = -1;
        for(int j = 0; j < 10; j++){
            if(curr_output[0][j] > max_value){
                max_value = curr_output[0][j];
                index = j;
            }
        }

        OUT << i + 1 << "," << index << '\n';
    }

    OUT.close();
}



int main(){
    parse_training_data();
    cerr << "Training data loaded. Total examples: " << train_input.size() << endl;
    
    try {
        train(); // Call your training function
    } catch(exception& e) {
        cerr << "Exception during training: " << e.what() << endl;
    }

    test(); 

    cerr << "Training completed." << endl;
    return 0;
}
