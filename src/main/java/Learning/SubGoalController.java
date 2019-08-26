package Learning;

import Learning.DeepQ.DeepQLearner;
import Learning.DeepQ.MLP;
import Model.Agent;
import Model.Simulation;
import Navigation.OrthogonalSubGoals;
import View.MainFrame;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public abstract class SubGoalController implements Serializable {

    private final int max_runs = 20;
    private int run=0;
    protected static int iter=0;
    private static int nrErrors = 0;
    protected int iterations = 500;
    protected float explorationRate;
    protected float exploreDiscount = explorationRate/iterations;

    private OrthogonalSubGoals subGoals;
    private Simulation model;

    //Variables needed for debugging:
    protected final static boolean use_gui = false;
    protected final static boolean debugging = false;
    protected final static int timeActionShown = 250;
    protected int showActionFor;
    protected long randSeed = 0;
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
    protected String algorithm = "Bresenham";
    protected Fitness fit;
    protected Features f;
    private int lowestCost;
    private int[][] costArr;
    private Random rand;

    protected HashSet<String> assignedGoals;
    protected HashMap<Agent, HashMap<String, List<IndexActLink>>> subGoalActivation;
    protected final List<String> subGoalKeys = Arrays.asList(new String[]{"WW", "SW", "SS", "SE", "EE", "NE", "NN", "NW"});

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
//
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

    public SubGoalController(){
        f = new Features();
        fit = new Fitness();
        rand = new Random();
        this.model = new Simulation(false);

        // init subGoals
        double fireLocation[] = f.locationCenterFireAndMinMax(model);
        subGoals = new OrthogonalSubGoals((int)fireLocation[0],(int)fireLocation[1], distMap, algorithm, model.getAllCells());

        assignedGoals = new HashSet<>();

        while (run<max_runs) {
            run++;
            System.out.println("=====================STARTING RUN #" + run + "==================");
            lowestCost = Integer.MAX_VALUE;
            costArr = new int[iterations][3];
            randSeed = 0;
            explorationRate = 0.3f;


            initRL();

            for (iter = 0; iter < iterations; iter++) {
                randSeed++;
                showActionFor = timeActionShown;
                train();
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



    public void updateDistMap(String key, Agent agent, List<IndexActLink> list){

        if (!(subGoals.isGoalOfAgent(key)||assignedGoals.contains(key))) {
//            if (debugging) {
//                System.out.println("updating goal " + key + " for agent #" + agent.getId());
//            }
            double[] input = getInputSet(key, agent);

            setDistance(list, key);
//
//            System.out.println(key + " " + goalToCostMap.keySet().contains(key) + " size map: " + goalToCostMap.keySet().size());
            goalToCostMap.get(key).setStateX(input);
//                double[] in = goalToCostMap.get(key).stateX;
//                if (debugging) {
//                    System.out.println("Input changed to " + Arrays.toString(goalToCostMap.get(key).stateX) + " from " + Arrays.toString(in));
//                }

        } else {
            if (debugging) {
                System.out.println("Not updating goal already assigned to/reached by other agent -> ");
            }
        }
    }

    public void updateDistMap(HashMap<Agent, HashMap<String, List<IndexActLink>>> subGoalOrder){
        for (Agent a:model.getAgents()){
            for (String key : distMap.keySet()){
                updateDistMap(key, a, subGoalOrder.get(a).get(key));
            }
            subGoals.selectClosestSubGoal(a);
            goalToCostMap.get(subGoals.getAgentGoals().get(a)).setStateX(getInputSet(subGoals.getNextGoal(a),a));
        }
    }

//    public void takeNextAction(Agent a) {
//        if (model.goalsHit<distMap.keySet().size()) {
//            if (a.onGoal()) {
//                assignedGoals.add(subGoals.getAgentGoals().get(a));
//                updateGoalsHit(a);
//                updateDistMap(subGoals.getNextGoal(a), a);
//                subGoals.setNextGoal(a);
//            }
//            String nextAction = a.subGoal.getNextAction();
//            if (nextAction.equals("PathFailed")){
//                subGoals.resetGoal(a);
//                takeNextAction(a);
//                return;
//            } else {
//                a.takeAction(nextAction);
//            }
//            // TODO: This piece of code is ugly as hell, come up with better solution
//            if (model.getAllCells().get(a.getX()).get(a.getY()).isBurning()) {
//                subGoals.removeGoalReached(a);
//                backup = a;
//                if (debugging) {
//                    System.out.println("Nr of Agents: " + model.getAgents().size());
//                }
//            }
//        } else { //Once all goals have been reached, the agent should stop moving as there is no use for it anymore.
//            a.takeAction("Do Nothing");
//        }
//    }

    public String getNextAction(Agent a){
        if (model.goalsHit<distMap.keySet().size()) {
            if (a.onGoal()) {
                assignedGoals.add(subGoals.getAgentGoals().get(a));
                updateGoalsHit(a);

                return "Update sub-goal";
//                updateDistMap(subGoals.getNextGoal(a), a);
//                subGoals.setNextGoal(a);
            }
            String nextAction = a.subGoal.getNextAction();
            if (nextAction.equals("PathFailed")){
                subGoals.resetGoal(a);
                return getNextAction(a);
            } else {
                return  nextAction;
            }
            // TODO: This piece of code is ugly as hell, come up with better solution
//            if (model.getAllCells().get(a.getX()).get(a.getY()).isBurning()) {
//                subGoals.removeGoalReached(a);
//                backup = a;
//                if (debugging) {
//                    System.out.println("Nr of Agents: " + model.getAgents().size());
//                }
//            }
        } else { //Once all goals have been reached, the agent should stop moving as there is no use for it anymore.
            return "Do Nothing";
        }
    }

    public void pickNextAction(Agent a){
        String action = getNextAction(a);
        if (action.equals("Update sub-goal")){
            String nextGoal = getNextGoal(a);
            double[] activation = getOutput(getInputSet(nextGoal,a));
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

    public void removeGoalReached(Agent a){
        subGoals.removeGoalReached(a);
    }


    public void setDistance(List<IndexActLink> activationList, String key) {
        int i = 0;
           do {
              subGoals.updateSubGoal(key, activationList.get(i).getIndex());
              i++;
              if (debugging){
                  if (i>7){
                      System.out.println("ACTIVATIONLIST > 7 : " + activationList.size());
                  }
              }
              if (i>=activationList.size()){//TODO: Somehow reset the simulation
//                  resetSimulation("All locations invalid");
              }
            } while (!subGoals.checkSubGoal(key, model.getAgents()));
        if (use_gui && (debugging)) {
            subGoals.paintGoal(key);
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
     * This is where to e-greedy search is implemented. The passed list is ordered according to activation in (1-e) of the cases
     * In the other cases, a randomized list with replacement is returned.
     * @param activation The activation according to which the list needs to be order that needs to ordered
     * @return Ordered list
     */
    public List<IndexActLink> determineOrder(double[] activation, float epsilon){
        List<IndexActLink> outputList = new LinkedList<>();

        //For regular greedy-search, the index which had the lowest activation will be placed at the first location of list
        for (int i = 0; i<activation.length; i++){
            outputList.add(new IndexActLink(i, (float) activation[i]));
        }

        //For the e-part of the search, there is an (1-e) chance that all indexes are randomized. no need to order it afterwards
        if (rand.nextFloat()<epsilon){
            for (int i = 0; i<activation.length; i++){
                outputList.get(i).index=rand.nextInt(activation.length);
            }
        } else {
            outputList.sort(Comparator.comparing(IndexActLink::getActivation, Comparator.nullsLast(Comparator.naturalOrder())));
        }

        return outputList;
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
        f.dispose();
    }

    protected void sleep(int t){
        try {
            Thread.sleep(Math.abs(t));
        } catch (java.lang.InterruptedException e) {
            System.out.println(e.getMessage());
        }
    }

    protected void screenshot(int run, int i){
        Rectangle screenRect = new Rectangle(Toolkit.getDefaultToolkit().getScreenSize());
        try {
            BufferedImage capture = new Robot().createScreenCapture(screenRect);
            ImageIO.write(capture, "bmp", new File("./screenshot_run"+ run +"_i_"+i+".bmp"));

        }catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
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

    private void updateGoalsHit(Agent agent){
        if (agent.isCutting()){
            model.goalsHit++;
            goalToCostMap.get(subGoals.getAgentGoals().get(agent)).setStateXPrime(getInputSet(subGoals.getNextGoal(agent),agent));
        }
        if (debugging){
            System.out.println("# of goals hit: " + model.goalsHit);
        }
    }

//    private void resetSimulation(String error){
//        System.out.println("UNEXPECTED ERROR: (" + error + ") OCCURRED, DISCARDING CURRENT MODEL AND STARTING NEW");
//        nrErrors++;
//        System.out.println("Distance Map: " + Collections.singletonList(distMap));
//        takeScreenShot();
//        trainMLP();
//    }

    private String dirGenerator(){
        return System.getProperty("user.dir") + "/results/Q-Learning/" + algorithm + "/" + model.getNr_agents() + "_agent_environment";
    }

//    private void writePerformanceFile(){
//        String dir = dirGenerator();
//        File file = new File(dir);
//        if (file.mkdirs() || file.isDirectory()) {
//            try {
//                FileWriter csvWriter = new FileWriter(dir + "/run" + run + ".csv");
//                csvWriter.append("Iteration");
//                csvWriter.append(",");
//                csvWriter.append("BurnCost");
//                csvWriter.append(",");
//                csvWriter.append("MoveCost");
//                csvWriter.append(",");
//                csvWriter.append("AgentDeathPenalty");
//                csvWriter.append("\n");
//
//                for (int i = 0; i< iterations; i++){
//                    csvWriter.append(i+","+costArr[i][0]+","+costArr[i][1]+","+costArr[i][2]+"\n");
//                }
//
//                csvWriter.flush();
//                csvWriter.close();
//            } catch (IOException e) {
//                System.out.println("Some IO-exception occurred");
//                e.printStackTrace();
//            }
//        } else {
//            System.out.println("Unable to make directory");
//        }
//    }
//
    public void printFinalDistMap(){
        System.out.println("Final distance Map: " + Collections.singletonList(distMap));
        System.out.println("Final number of sub goals assigned: " + goalToCostMap.keySet().size());
        for (Map.Entry<String, InputCost> entry : goalToCostMap.entrySet()) {
            System.out.println(entry.getKey()+" : "+Arrays.toString(entry.getValue().stateX)+" -> "+Arrays.toString(entry.getValue().stateXPrime));
        }
    }

    protected abstract double[] getOutput(double[] input);

    protected abstract void train();

    protected abstract void initRL();

    public double getDist(String key){
        return distMap.get(key);
    }

    public String getNextGoal(Agent a){
        return subGoals.getNextGoal(a);
    }

    public void setNextGoal(Agent a){
        subGoals.setNextGoal(a);
    }

    public Map<String, InputCost> getGoalToCostMap() {
        return goalToCostMap;
    }

    //
//    /**
//     * Class needed to order a list containing the index of the distance and the activation of that distance.
//     */
//    public class IndexActLink{
//        private int index;
//        private float activation;
//
//        private IndexActLink(int i, float a){
//            index=i;
//            activation=a;
//        }
//
//        private double getActivation() {
//            return activation;
//        }
//
//        public int getIndex() {
//            return index;
//        }
//    }
//
    public class InputCost{
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

        public void setCost(int cost){
            this.cost = cost;
        }

        public double[] getStateX() {
            return stateX;
        }

        public double[] getStateXPrime() {
            return stateXPrime;
        }

        public int getCost() {
            return cost;
        }
    }

    /**
     * Class needed to order a list containing the index of the distance and the activation of that distance.
     */
    public class IndexActLink{
        private int index;
        private float activation;

        private IndexActLink(int i, float a){
            index=i;
            activation=a;
        }

        public double getActivation() {
            return activation;
        }

        public int getIndex() {
            return index;
        }
    }

}
