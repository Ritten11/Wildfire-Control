package Learning.DeepQ;

import Learning.Features;
import Learning.Fitness;
import Learning.RLController;
import Model.Agent;
import Model.Simulation;
import Navigation.OrthogonalSubGoals;
import View.MainFrame;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;


public class DeepQLearner implements RLController, Serializable {
    private final int max_runs = 20;
    private int run=0;
    private static int iter=0;
    private static int nrErrors = 0;
    protected int iterations = 2000;
    protected float explorationRate = 0.10f;
    protected float exploreDiscount = explorationRate/iterations;
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
    private OrthogonalSubGoals subGoals;
    private Simulation model;

    //Variables needed for debugging:
    final static boolean use_gui = true;
    final static boolean debugging = false;
    private final static int timeActionShown = 250;
    private int showActionFor;
    private long randSeed = 0;
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
    private String algorithm = "Bresenham";
    private Fitness fit;
    private Features f;
    private int lowestCost;
    private int[][] costArr;

    private HashSet<String> assignedGoals;

    private Map<String,Double> distMap = Stream.of(
            new AbstractMap.SimpleEntry<>("WW", 0.0),
            new AbstractMap.SimpleEntry<>("SW", 0.0),
            new AbstractMap.SimpleEntry<>("SS", 0.0),
            new AbstractMap.SimpleEntry<>("SE", 0.0),
            new AbstractMap.SimpleEntry<>("EE", 0.0),
            new AbstractMap.SimpleEntry<>("NE", 0.0),
            new AbstractMap.SimpleEntry<>("NN", 0.0),
            new AbstractMap.SimpleEntry<>("NW", 0.0))
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

    private Map<String, InputCost> goalToCostMap= Stream.of(
            new AbstractMap.SimpleEntry<>("WW", new InputCost()),
            new AbstractMap.SimpleEntry<>("SW", new InputCost()),
            new AbstractMap.SimpleEntry<>("SS", new InputCost()),
            new AbstractMap.SimpleEntry<>("SE", new InputCost()),
            new AbstractMap.SimpleEntry<>("EE", new InputCost()),
            new AbstractMap.SimpleEntry<>("NE", new InputCost()),
            new AbstractMap.SimpleEntry<>("NN", new InputCost()),
            new AbstractMap.SimpleEntry<>("NW", new InputCost()))
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

    public DeepQLearner(){
        f = new Features();
        fit = new Fitness();
        rand = new Random();
        while (run<max_runs) {
            run++;
            System.out.println("=====================STARTING RUN #" + run + "==================");
            lowestCost = Integer.MAX_VALUE;
            costArr = new int[iterations][3];
            randSeed = 0;
            explorationRate = 0.1f;


            initNN();

            for (iter = 0; iter < iterations; iter++) {
                randSeed++;
                showActionFor = timeActionShown;
                trainMLP();
                costArr[iter] = getCost();
                if (explorationRate > 0) {
                    explorationRate -= exploreDiscount;
                }
            }

            writePerformanceFile();

            if (nrErrors != 0) {
                System.out.println("Total # of errors occurred: " + nrErrors);
            }
        }

    }

    private void trainMLP(){
        model = new Simulation(this, use_gui, randSeed);
        backup = model.getAgents().get(0);
        assignedGoals = new HashSet<>();

        double fireLocation[] = f.locationCenterFireAndMinMax(model);
        subGoals = new OrthogonalSubGoals((int)fireLocation[0],(int)fireLocation[1], distMap, algorithm, model.getAllCells());

        JFrame frame;
        if(use_gui){
            frame = createMainFrame();

        }


        for (Agent a:model.getAgents()){
//            if (debugging){
//                System.out.println("assigning goal to agent: " + a.getId());
//            }
            updateDistMap(a);
            subGoals.selectClosestSubGoal(a);
        }


        if ((debugging)&&use_gui){
            //model.applyUpdates();
            sleep(500);
            takeScreenShot();
        }

        model.start();
        if (debugging){
            System.out.println("Final distance Map: " + Collections.singletonList(distMap));
            System.out.println("Final number of sub goals assigned: " + goalToCostMap.keySet().size());
            for (Map.Entry<String, InputCost> entry : goalToCostMap.entrySet()) {
                System.out.println(entry.getKey()+" : "+Arrays.toString(entry.getValue().stateX)+" -> "+Arrays.toString(entry.getValue().stateXPrime));
            }
        }
        int[] costArr = getCost();
        int cost = costArr[0] + costArr[1] + costArr[2];
        for (InputCost inputCost:goalToCostMap.values()){
            inputCost.setCost(cost);
        }

        if (cost<lowestCost){
            lowestCost = cost;
            System.out.println("In iteration: " + iter + " a solution was found with cost: " + lowestCost);
        }

        if (model.getAgents().isEmpty()){ //TODO: Again, should really try to come up with better solution.
            model.getAgents().add(backup);
        }

        for (Map.Entry<String, InputCost> entry: goalToCostMap.entrySet()){
            String key = entry.getKey();
            InputCost ic = entry.getValue();
            if (ic.stateXPrime==null){ //If the goal has not been reached, set the result of that goal to the default feature vector
                ic.setStateXPrime(getInputSet("WW", model.getAgents().get(0)));
                System.out.println("updated next state of " + key + " to: " + Arrays.toString(ic.stateXPrime));
            }
            double action = distMap.get(key);
            train(ic.stateX, ic.stateXPrime, Math.toIntExact(Math.round(action)), ic.cost);

        }

        if (use_gui){
            disposeMainFrame(frame);
        }
    }

    private void train(double[] oldState, double[] newState, int action, int reward){

        double[] oldValue = getQ(oldState);

//        System.out.println(Arrays.toString(oldState)+" -> " +Arrays.toString(oldValue));

        double[] newValue = getQ(newState);

        oldValue[action] = reward + gamma* minValue(newValue);

        double[] trainInput = oldState;
        double[] trainOutput = oldValue;

        addToInputBatch(trainInput);
        addToOutputBatch(trainOutput);

        batchNr++;

        if (batchNr%batchSize==0){
//            System.out.println("Updating MLP");
//            if (debugging){
//                System.out.println("For action: " + action);
//            }
            batchNr = 0;
            mlp.updateMLP(inputBatch, outputBatch);
        }

        oldValue = getQ(oldState);
//        System.out.println(Arrays.toString(oldState)+" -> "+Arrays.toString(oldValue));
    }

    private void updateDistMap(String key, Agent agent){

        if (!(subGoals.isGoalOfAgent(key)||assignedGoals.contains(key))) {
//            if (debugging) {
//                System.out.println("updating goal " + key + " for agent #" + agent.getId());
//            }
            double[] input = getInputSet(key, agent);

            setDistance(input, key);
//
//            System.out.println(key + " " + goalToCostMap.keySet().contains(key) + " size map: " + goalToCostMap.keySet().size());
            goalToCostMap.get(key).setStateX(input);
//                double[] in = goalToCostMap.get(key).stateX;
//                if (debugging) {
//                    System.out.println("Input changed to " + Arrays.toString(goalToCostMap.get(key).stateX) + " from " + Arrays.toString(in));
//                }

        } else {
//            if (debugging) {
//                System.out.println("Not updating goal already assigned to/reached by other agent -> ");
//            }
        }
    }

    private void updateDistMap(Agent a){
        for (String key : distMap.keySet()){
            updateDistMap(key, a);
        }
    }

    @Override
    public void pickAction(Agent a) {
        if (model.goalsHit<distMap.keySet().size()) {
            if (a.onGoal()) {
                assignedGoals.add(subGoals.getAgentGoals().get(a));
                updateGoalsHit(a);
                updateDistMap(subGoals.getNextGoal(a), a);
                subGoals.setNextGoal(a);
            }
            String nextAction = a.subGoal.getNextAction();
            if (nextAction.equals("PathFailed")){
                subGoals.resetGoal(a);
                pickAction(a);
                return;
            } else {
                a.takeAction(nextAction);
            }
            // TODO: This piece of code is ugly as hell, come up with better solution
            if (model.getAllCells().get(a.getX()).get(a.getY()).isBurning()) {
                subGoals.removeGoalReached(a);
                backup = a;
                if (debugging) {
                    System.out.println("Nr of Agents: " + model.getAgents().size());
                }
            }

            if (use_gui) {
                if (showActionFor > 0) {
                    sleep(showActionFor);
                    showActionFor -= 0;
                }
            }
        } else { //Once all goals have been reached, the agent should stop moving as there is no use for it anymore.
            a.takeAction("Do Nothing");
        }

        // TODO: Check if other RL techniques have similar kill command
//        if (model.getActiveCells().size()==0){
//            model.stop("No more active cells");
//        }

    }

    private void initNN(){
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



    protected List<IndexActLink> greedyLocation(double[] state){
        double[] outputSet = getQ(state);

//        if (debugging) {
//            System.out.println("New activation output: " + Arrays.toString(outputSet));
//        }
        List<IndexActLink> outputList = new LinkedList<>();

        for (int i = 0; i<outputSet.length; i++){
            outputList.add(new IndexActLink(i, (float) outputSet[i]));
        }

        outputList.sort(Comparator.comparing(IndexActLink::getActivation, Comparator.nullsLast(Comparator.naturalOrder())));

//        if (debugging){
//            System.out.println("Optimal option: " + outputList.get(0).index);
//        }
        return outputList;
    }

    protected int randomLocation(){
        return rand.nextInt(sizeOutput);
    }

    protected int boltzmannDistAct(List<IndexActLink> activationList){
        double sum = 0;
        for (IndexActLink ial:activationList){
            sum += Math.exp(1/(explorationRate*ial.activation));
        }
        if (debugging){
            for (IndexActLink ial:activationList){
                System.out.println("Sum: " + sum + " -> index #" + ial.index + " has prob of: " + Math.exp(1/(explorationRate*ial.activation))/sum);
            }
        }
        if (Double.isInfinite(sum)){
            sum = Double.MAX_VALUE;
            System.out.println("infinite value");
        }
        double randDouble = rand.nextDouble();
        double randActionSum = Math.exp(1/(explorationRate*activationList.get(0).activation))/sum;
        int i = 0;
        while (randDouble>randActionSum){
            i++;
            randActionSum+= Math.exp(1/(explorationRate*activationList.get(0).activation))/sum;
        }
        if (debugging){
            System.out.println("Returning " + activationList.get(i).index + " as randDouble = " + randDouble + " and randActionsum  = " + randActionSum);
        }
        return activationList.get(i).index;
    }

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

    public void setDistance(double in[], String key) {
        float randFloat = rand.nextFloat();
        int i = 0;
//        int actionIndex;
        List<IndexActLink> activationList = greedyLocation(in);
//        do { //TODO: Implement Boltzmann-distributed Exploration: https://www.researchgate.net/publication/2502531_The_Role_Of_Exploration_In_Learning_Control
//            actionIndex = boltzmannDistAct(activationList);
//            subGoals.updateSubGoal(key,actionIndex);
//
//            Iterator<IndexActLink> iter = activationList.iterator();
//            while(iter.hasNext()) {
//                IndexActLink ial = iter.next();
//                if (ial.index == actionIndex) {
//                    iter.remove();
//                }
//            }
//
//        } while (!subGoals.checkSubGoal(key, model.getAgents())&&!activationList.isEmpty());
        if (randFloat > explorationRate) {
            do {
                subGoals.updateSubGoal(key, activationList.get(i).index);
                i++;
                if (debugging){
                    if (i>7){
                        System.out.println("ACTIVATIONLIST > 7 : " + activationList.size());
                    }
                }
                if (i>=activationList.size()){
                    resetSimulation("All locations invalid");
                }

            } while (!subGoals.checkSubGoal(key, model.getAgents()));
        } else {
            do {
                if (i>=10){
                    resetSimulation("Could not find suitable random location");
                }
                subGoals.updateSubGoal(key, randomLocation());
                i++;
            } while (!subGoals.checkSubGoal(key, model.getAgents()));
        }
        if (use_gui && (debugging)) {
            subGoals.paintGoal(key);
        }
    }

    private int[] getCost(){
        int cost[]=new int[3];
        cost[0]=fit.totalFuelBurnt(model);
        cost[1]=fit.totalMoveCost(model);
        cost[2]=fit.totalAgentPenalty(model);
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
        float windX = model.getParameters().get("Wind x");
        float windY = model.getParameters().get("Wind y");
        float windS = model.getParameters().get("Wind Speed");
        double[] set = f.appendDoubles(f.cornerVectors(model, a, false), f.windRelativeToSubgoal(windX, windY, subGoal)*windS, f.numberSubGoalsHit(model));
        //double[] set = new double[]{f.windRelativeToSubgoal(windX, windY, subGoal)};
        return set;
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

    protected void takeScreenShot(){
        JFrame f = createMainFrame();
        sleep(500);
        screenshot(subGoals.getGoalsReached().size(), lowestCost);
        f.dispose();
    }

    protected JFrame createMainFrame(){
        JFrame f = new MainFrame(model);
        sleep(1000);
        return f;
    }

    protected void disposeMainFrame(JFrame f){
        sleep(500);
        //f.dispose();
    }

    protected void sleep(int t){
        try {
            Thread.sleep(Math.abs(t));
        } catch (java.lang.InterruptedException e) {
            System.out.println(e.getMessage());
        }
    }

    protected void screenshot(int goalsHit, int i){
        Rectangle screenRect = new Rectangle(Toolkit.getDefaultToolkit().getScreenSize());
        try {
            BufferedImage capture = new Robot().createScreenCapture(screenRect);
            ImageIO.write(capture, "bmp", new File("./screenshot_run"+ run +"_i_"+iter+".bmp"));

        }catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
    }

    private void updateGoalsHit(Agent agent){
        if (agent.isCutting()){
            model.goalsHit++;
            goalToCostMap.get(subGoals.getAgentGoals().get(agent)).setStateXPrime(getInputSet(subGoals.getNextGoal(agent),agent));
        }
        if (debugging){
            System.out.println("# of goals hit: " + model.goalsHit);
        }
    }

    private void resetSimulation(String error){
        System.out.println("UNEXPECTED ERROR: (" + error + ") OCCURRED, DISCARDING CURRENT MODEL AND STARTING NEW");
        nrErrors++;
        System.out.println("Distance Map: " + Collections.singletonList(distMap));
        takeScreenShot();
        trainMLP();
    }

    private String dirGenerator(){
        return System.getProperty("user.dir") + "/results/Q-Learning/" + algorithm + "/" + model.getNr_agents() + "_agent_environment";
    }

    private void writePerformanceFile(){
        String dir = dirGenerator();
        File file = new File(dir);
        if (file.mkdirs() || file.isDirectory()) {
            try {
                FileWriter csvWriter = new FileWriter(dir + "/run" + run + ".csv");
                csvWriter.append("Iteration");
                csvWriter.append(",");
                csvWriter.append("BurnCost");
                csvWriter.append(",");
                csvWriter.append("MoveCost");
                csvWriter.append(",");
                csvWriter.append("AgentDeathPenalty");
                csvWriter.append("\n");

                for (int i = 0; i< iterations; i++){
                    csvWriter.append(i+","+costArr[i][0]+","+costArr[i][1]+","+costArr[i][2]+"\n");
                }

                csvWriter.flush();
                csvWriter.close();
            } catch (IOException e) {
                System.out.println("Some IO-exception occurred");
                e.printStackTrace();
            }
        } else {
            System.out.println("Unable to make directory");
        }
    }

    /**
     * Class needed to order a list containing the index of the distance and the activation of that distance.
     */
    private class IndexActLink{
        private int index;
        private float activation;

        private IndexActLink(int i, float a){
            index=i;
            activation=a;
        }

        private double getActivation() {
            return activation;
        }

        public int getIndex() {
            return index;
        }
    }

    class InputCost{
        double[] stateX;
        double[] stateXPrime;
        int cost;

        private InputCost(){}

        public void setStateX(double[] stateX){
            this.stateX = stateX;
        }

        public void setStateXPrime(double[] stateXPrime) {
            this.stateXPrime = stateXPrime;
        }

        private void setCost(int cost){
            this.cost = cost;
        }
    }
}
