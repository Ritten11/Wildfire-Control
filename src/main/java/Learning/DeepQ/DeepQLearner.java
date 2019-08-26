package Learning.DeepQ;

import Learning.RLController;
import Learning.SubGoalController;
import Model.Agent;
import Model.Simulation;
import View.MainFrame;

import javax.swing.*;
import java.io.*;
import java.util.*;
import java.util.List;


/**
 * Function of this class: Produce an array or HashSet containing the distances between subgoals and center of fire
 * Not the function of this class: subgoal management.
 */
public class DeepQLearner extends SubGoalController implements RLController, Serializable {
    protected float gamma = 0.1f;
    protected float alpha = 0.001f;


    //parameters for neural network construction.
    private int sizeInput;
    private int sizeOutput;
    private int nrHidden; //TODO: create compatibility for dynamic number of hidden layers
    private int sizeHidden = 50;
    private int batchSize = 1;

    private int batchNr;
    private double inputBatch[][];
    private double outputBatch[][];

    private MLP mlp;
    private Random rand;
//    private OrthogonalSubGoals subGoals;
    private Simulation model;

    //Variables needed for debugging:
    /* IF RANDOM SEED IN SIMULATION CLASS == 0
     0 -> d=0
     1 -> d=5
     2 -> d=0
     3 -> d=4
     4 -> d=8
     5 -> d=2 (if the fire is sufficiently far away)
     6 -> d=5
     6 -> d=6
     */

    private Agent backup; //If the final agent has died, the MLP still needs an agent to determine the inputVector of the MLP

    //Fields for functionality of navigation and fitness
    private int lowestCost;
    private int[][] costArr;


    public DeepQLearner(){
        super();
    }

    protected void train(){
        model = new Simulation(this, use_gui, randSeed);
        //SGC = new SubGoalController(algorithm, "CQL", model, rand,  use_gui, debugging);
        subGoalActivation = new HashMap<>();
        backup = model.getAgents().get(0);
//        assignedGoals = new HashSet<>();

//        double fireLocation[] = f.locationCenterFireAndMinMax(model);
//        subGoals = new OrthogonalSubGoals((int)fireLocation[0],(int)fireLocation[1], distMap, algorithm, model.getAllCells());

        JFrame frame;
        if(use_gui){
            frame = createMainFrame();

        }



        initSubGoalOrder();
        updateDistMap(subGoalActivation);

        if ((debugging)&&use_gui){
            //model.applyUpdates();
            sleep(500);
            //SGC.screenshot(run, iter);
        }

        model.start();
        if (debugging){
            printFinalDistMap();
           // printGoalToCoastMap();
        }
        int[] costArr = getCost();
        int cost = costArr[0] + costArr[1] + costArr[2];
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

        if (use_gui){
            disposeMainFrame(frame);
        }
    }

    private void initSubGoalOrder(){ //TODO: If rand float< exploreRate -> make random dist array, otherwise use order of activations
        for (Agent a:model.getAgents()){
            HashMap<String, List<IndexActLink>> activationMap = new HashMap<>();
            for (String s:subGoalKeys){
                double [] outputSet = getQ(getInputSet(s, a));

                List<IndexActLink> outputList = determineOrder(outputSet, explorationRate);

                activationMap.put(s, outputList);
            }
            subGoalActivation.put(a, activationMap);
        }
        if (debugging) {
            printSubGoalActivation();
        }
    }





    private void printSubGoalActivation(){
        for (Agent a:model.getAgents()){
            HashMap<String, List<IndexActLink>> act = subGoalActivation.get(a);
            System.out.println("Sub-goals of agent #" + a.getId());
            for (String s:subGoalKeys){
                System.out.println("\nActivation of sub-goal " + s + ": ");
                for (IndexActLink ial : act.get(s)) {
                    System.out.print( ial.getActivation() + "(" + ial.getIndex()  + ") " );
                }

            }
            System.out.println(" ");
        }
    }

    private void train(double[] oldState, double[] newState, int action, int reward){

        double[] oldValue = getQ(oldState);

        System.out.println(Arrays.toString(oldState)+" -> " +Arrays.toString(oldValue));

        double[] newValue = getQ(newState);

        oldValue[action] = reward + gamma* minValue(newValue);

        double[] trainInput = oldState;
        double[] trainOutput = oldValue;

        addToInputBatch(trainInput);
        addToOutputBatch(trainOutput);

        batchNr++;

        if (batchNr%batchSize==0){
            System.out.println("Updating MLP");
            if (debugging){
                System.out.println("For action: " + action);
            }
            batchNr = 0;
            mlp.updateMLP(inputBatch, outputBatch);
        }

        oldValue = getQ(oldState);
        System.out.println(Arrays.toString(oldState)+" -> "+Arrays.toString(oldValue));
    }

    @Override
    public void pickAction(Agent a) {
        String action = getNextAction(a);
        if (action.equals("Update sub-goal")){
            String nextGoal = getNextGoal(a);
            double[] activation = getQ(getInputSet(nextGoal,a));
            updateDistMap(nextGoal, a, determineOrder(activation, explorationRate));
            setNextGoal(a);
            action = getNextAction(a);
        }
        a.takeAction(action);
        if (model.getAllCells().get(a.getX()).get(a.getY()).isBurning()) {
            removeGoalReached(a);
            backup = a;
            if (debugging) {
                System.out.println("Nr of Agents: " + model.getAgents().size());
            }
        }
        if (use_gui) {
            if (showActionFor > 0) {
                sleep(showActionFor);
                showActionFor -= 1;
            }
        }
    }

    protected void initRL(){
        model = new Simulation(this);

        double[] fire=f.locationCenterFireAndMinMax(model);
        int minY=(int)Math.min(fire[1], (model.getAllCells().get(0).size()-fire[1]));
        int minX=(int)Math.min(fire[0], (model.getAllCells().size()-fire[0]));
        sizeOutput = Math.min(minX,minY);
        sizeInput = getInputSet("WW", model.getAgents().get(0)).length;

        batchNr = 0;
        inputBatch = new double[batchSize][sizeInput];
        outputBatch = new double[batchSize][sizeOutput];

        mlp = new MLP(sizeInput, sizeHidden, sizeOutput, alpha, batchSize, new Random().nextLong());
    }
//
//    protected int boltzmannDistAct(List<IndexActLink> activationList){
//        double sum = 0;
//        for (IndexActLink ial:activationList){
//            sum += Math.exp(1/(explorationRate*ial.activation));
//        }
//        if (debugging){
//            for (IndexActLink ial:activationList){
//                System.out.println("Sum: " + sum + " -> index #" + ial.index + " has prob of: " + Math.exp(1/(explorationRate*ial.activation))/sum);
//            }
//        }
//        if (Double.isInfinite(sum)){
//            sum = Double.MAX_VALUE;
//            System.out.println("infinite value");
//        }
//        double randDouble = rand.nextDouble();
//        double randActionSum = Math.exp(1/(explorationRate*activationList.get(0).activation))/sum;
//        int i = 0;
//        while (randDouble>randActionSum){
//            i++;
//            randActionSum+= Math.exp(1/(explorationRate*activationList.get(0).activation))/sum;
//        }
//        if (debugging){
//            System.out.println("Returning " + activationList.get(i).index + " as randDouble = " + randDouble + " and randActionsum  = " + randActionSum);
//        }
//        return activationList.get(i).index;
//    }

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



    private int[] getCost(){
        int[] cost = fit.totalCosts(model);
        if (debugging){
            System.out.println("Total fuel burnt: " + fit.totalFuelBurnt(model) + ", Total moveCost: " + fit.totalMoveCost(model) + ", Total cost: " + (cost[0]+cost[1]+cost[2]));
        }
        return cost;
    }


    /**
     * Transform state to input vector.
     * @param subGoal: expressed as an integer to allow for use of for loops.
     * @return
     */
//    private double[] getInputSet(int subGoal){
//        float windX = model.getParameters().get("Wind x");
//        float windY = model.getParameters().get("Wind y");
//        double[] set = f.appendArrays(f.cornerVectors(model, false), f.windRelativeToSubgoal(windX, windY, indexMap.get(subGoal)));
//        return set;
//    }

    private double[] getInputSet(String subGoal, Agent a){
        return f.getInputSet(model, a, subGoal);
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

    protected JFrame createMainFrame(){
        JFrame f = new MainFrame(model);
        sleep(1000);
        return f;
    }

    protected void disposeMainFrame(JFrame f){
        sleep(500);
        f.dispose();
    }
//
//    private void updateGoalsHit(Agent agent){
//        if (agent.isCutting()){
//            model.goalsHit++;
//            goalToCostMap.get(subGoals.getAgentGoals().get(agent)).setStateXPrime(getInputSet(subGoals.getNextGoal(agent),agent));
//        }
//        if (debugging){
//            System.out.println("# of goals hit: " + model.goalsHit);
//        }
//    }
//
//    private void resetSimulation(String error){
//        System.out.println("UNEXPECTED ERROR: (" + error + ") OCCURRED, DISCARDING CURRENT MODEL AND STARTING NEW");
//        nrErrors++;
//        System.out.println("Distance Map: " + Collections.singletonList(distMap));
//        takeScreenShot();
//        train();
//    }

    private String dirGenerator(){
        return System.getProperty("user.dir") + "/results/Q-Learning/" + algorithm + "/" + model.getNr_agents() + "_agent_environment";
    }
//
//    private void printGoalToCoastMap(){
//        for (String s:goalToCostMap.keySet()){
//            System.out.println("Result for goal " + s + ": " + goalToCostMap.get(s).toString());
//        }
//    }
//
//    public class InputCost{
//        double[] stateX;
//        double[] stateXPrime;
//        int cost;
//
//        private InputCost(){}
//
//        public void setStateX(double[] stateX){
//            this.stateX = stateX;
//        }
//
//        public void setStateXPrime(double[] stateXPrime) {
//            this.stateXPrime = stateXPrime;
//        }
//
//        private void setCost(int cost){
//            this.cost = cost;
//        }
//
//        @Override
//        public String toString() {
//            return "InputCost{" +
//                    "stateX=" + Arrays.toString(stateX) +
//                    ", stateXPrime=" + Arrays.toString(stateXPrime) +
//                    ", cost=" + cost +
//                    '}';
//        }
//    }
}
