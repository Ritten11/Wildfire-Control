package Learning.DeepQ;

import Learning.RLController;
import Learning.SubGoalController;
import Model.Agent;
import Model.Simulation;
import View.MainFrame;
import org.apache.commons.lang3.SerializationUtils;

import javax.swing.*;
import java.io.*;
import java.util.*;
import java.util.List;


/**
 * Function of this class: Produce an array or HashSet containing the distances between subgoals and center of fire
 * Not the function of this class: subgoal management.
 */
public class DeepQLearner extends SubGoalController implements Serializable {
    protected float alpha = 0.001f;


    //parameters for neural network construction.
    private int sizeInput;
    private int sizeOutput;
    private int nrHidden; //TODO: create compatibility for dynamic number of hidden layers
    private int sizeHidden;
    private int batchSize = 1;

    private int batchNr;
    private double inputBatch[][];
    private double outputBatch[][];

    protected MLP mlp;
    private double lowestCost;
    private double[] costs;

    private List<MLP> savedMLPList;


    public DeepQLearner(int nrAgents){
        super(nrAgents);
    }

    /**
     * The main training function which starts the simulation and determines the total costs afterwards. It
     * initialises the back-propagation of the total costs through the MLP and, if necessary, makes a deep copy of
     * the MLP to use in the testing-phase
     * @param saveMLP   - boolean stating whether or not to save the MLP (only used in CMC implementation)
     * @param finalIter - boolean stating whether or not to save the MLP (only used in CoSyNE implementation)
     */
    protected void train(boolean saveMLP, boolean finalIter){
        JFrame frame;
        costs = new double[3];
        if(use_gui){
            frame = createMainFrame();
        }

        if ((debugging)&&use_gui){
            sleep(500);
        }

        model.start();
        if (debugging){
            printFinalDistMap();
        }
        int[] costArr = fit.totalCosts(model);
        double cost = costArr[0] + costArr[1] + costArr[2];
        costs[0] = costArr[0];
        costs[1] = costArr[1];
        costs[2] = costArr[2];
        for (InputCost inputCost:getGoalToCostMap().values()){
            inputCost.setCost(cost);
        }

        if (cost<lowestCost){
            lowestCost = cost;
            System.out.println("In iteration: " + iter + " a solution was found with cost: " + lowestCost);
        }

        if (model.getAgents().isEmpty()){
            model.getAgents().add(backup);
        }

        for (Map.Entry<String, InputCost> entry: getGoalToCostMap().entrySet()){
            String key = entry.getKey();
            InputCost ic = entry.getValue();
            if (ic.getStateXPrime()==null){ //If the goal has not been reached, set the result of that goal to the default feature vector
                ic.setStateXPrime(getInputSet("WW", model.getAgents().get(0)));
                if (debugging) {
                    System.out.println("updated next state of " + key + " to: " + Arrays.toString(ic.getStateXPrime()));
                }
            }
            double action = getDist(key);
            train(ic.getStateX(), ic.getStateXPrime(), Math.toIntExact(Math.round(action)), ic.getCost());

        }

        if (saveMLP){
            savedMLPList.add(deepCopyMLP(mlp));
        }

        if (use_gui){
            disposeMainFrame(frame);
        }
    }


    /**
     * Method for updating the MLP through back-propagation
     * @param oldState - The state at which the action is taken
     * @param newState - The resulting state once an action has finished (not used).
     * @param action   - The action taken in "oldState"
     * @param reward   - The reward (or return) value resulting form taking the action in "oldState".
     */
    private void train(double[] oldState, double[] newState, int action, double reward){

        double[] oldValue = getQ(oldState);


        oldValue[action] = reward;

        double[] trainInput = oldState;
        double[] trainOutput = oldValue;
        addToInputBatch(trainInput);
        addToOutputBatch(trainOutput);

        batchNr++;

        if (batchNr%batchSize==0){
            if (debugging){
                System.out.println("Updating MLP");
                System.out.println("For action: " + action);
            }
            batchNr = 0;
            mlp.updateMLP(inputBatch, outputBatch);
        }

        if (debugging) {
            oldValue = getQ(oldState);
            System.out.println(Arrays.toString(oldState) + " -> " + Arrays.toString(oldValue));
        }
    }

    /**
     * method for testing the MLP on the same map for a set amount of iterations (sizeFinalComparison). The sum of
     * the results is stored in the costs array and eventually averaged by dividing of the total number of iterations
     */
    public void test(){
        costs = new double[3];
        for(int i = 0; i < sizeFinalComparison; i++){
            resetSimulation();
            mlp = savedMLPList.get(i);
            model.start();

            int[] modelCosts = fit.totalCosts(model);
            costs[0] += modelCosts[0];
            costs[1] += modelCosts[1];
            costs[2] += modelCosts[2];

        }

        costs[0] /= sizeFinalComparison;
        costs[1] /= sizeFinalComparison;
        costs[2] /= sizeFinalComparison;
    }

    /**
     * In some occasions, the simulation needs to be reset to the rest.
     */
    @Override
    protected void resetSimulation(){
        model = new Simulation(this,use_gui,randSeed, nrAgents);
        resetSubGoals();
    }

    /**
     * Returns a deep copy of the given MLP
     * @param mlp - The MLP to be copied
     * @return copy of the MLP
     */
    private MLP deepCopyMLP(MLP mlp) {
        return  (MLP) SerializationUtils.clone(mlp);
    }

    /**
     * Initialises and MLP with 30 hidden units in a single layer.
     */
    protected void initRL(){
        savedMLPList = new ArrayList<>();

        double[] fire=f.locationCenterFireAndMinMax(model);
        int minY=(int)Math.min(fire[1], (model.getAllCells().get(0).size()-fire[1]));
        int minX=(int)Math.min(fire[0], (model.getAllCells().size()-fire[0]));
        sizeOutput = Math.min(minX,minY);
        sizeHidden = 30;
        sizeInput = getInputSet("WW", model.getAgents().get(0)).length;

        alpha = 0.05f;

        batchSize = 1;
        inputBatch = new double[batchSize][sizeInput];
        outputBatch = new double[batchSize][sizeOutput];

        mlp = new MLP(sizeInput, sizeHidden, sizeOutput, alpha, batchSize, new Random().nextLong());
    }

    /**
     * Receives the feature vector, runs this through the MLP and returns the resulting q-values.
     * @param in -feature vector
     * @return - resulting q-values
     */
    private double[] getQ(double[] in){

        double input[][] = new double[1][in.length];
        for (int i = 0; i<in.length; i++){
            input[0][i] = in[i];
        }

        double activation[][] = mlp.getOutput(input);
        double output[]= new double[activation.length];
        for (int i = 0; i<activation.length; i++){
            output[i] = activation[i][0];
        }
        return output;
    }

    protected double[] getOutput(double[] input){
        return getQ(input);
    }

    protected double[] getCost(){
        return costs;
    }


    /**
     * This is where to e-greedy search is implemented. The passed list is ordered according to activation in (1-e) of the cases
     * In the other cases, a randomized list with replacement is returned.
     * @param activation The activation according to which the list needs to be order that needs to ordered
     * @return Ordered list
     */
    @Override
    public List<IndexActLink> determineOrder(double[] activation){

        Random rand = new Random();

        //For regular greedy-search, the index which had the lowest activation will be placed at the first location of list
        List<IndexActLink> outputList = makeIndexActList(activation);

        //For the e-part of the search, there is an (e) chance that all indexes are randomized. no need to order it afterwards
        if (rand.nextFloat()<explorationRate){
            for (int i = 0; i<activation.length; i++){
                outputList.get(i).setIndex(rand.nextInt(activation.length));
            }
        } else {
            outputList.sort(Comparator.comparing(IndexActLink::getActivation, Comparator.nullsLast(Comparator.naturalOrder())));
        }

        return outputList;
    }

    private void addToInputBatch(double in[]){
        for (int i = 0; i<in.length; i++){
            inputBatch[batchNr][i] = in[i];
        }
    }
    private void addToOutputBatch(double out[]){
        for (int i = 0; i<out.length; i++){
            outputBatch[batchNr][i] = out[i];
        }
    }

    public static double minValue(double[] numbers){
        double min = Double.MAX_VALUE;
        for(int i = 1; i < numbers.length;i++)
        {
            if(numbers[i] < min)
            {
                min = numbers[i];
            }
        }
        return min;
    }

    protected String defRLMethod() {return "Q_learning";}

    protected JFrame createMainFrame(){
        JFrame f = new MainFrame(model);
        sleep(1000);
        return f;
    }

    protected void disposeMainFrame(JFrame f){
        sleep(500);
        f.dispose();
    }
}
