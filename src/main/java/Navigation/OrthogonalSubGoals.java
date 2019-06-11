package Navigation;

import Model.Agent;
import Model.Elements.Element;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

//TODO: ONLY COMPATIBLE WITH SINGLE AGENT!!!
public class OrthogonalSubGoals implements Serializable {
    List<List<Element>> cells;
    int fireX, fireY;
    String algorithm;
    Map<String,Double> distMap;

    private Map<Agent, String> agentGoals; //semi-solution for multi agent problem
    private String defaultKey = "WW"; //If no other goal can be selected resort to the default goal
    final int maxNrGoals = 8;

    private HashMap<String, Element> subGoals;


    //used for directions of subgoals
    int dx[]={-1,-1,0,1,1,1,0,-1};
    int dy[]={0,-1,-1,-1,0,1,1,1};

    //Might be usefull when linking the goal direction implementation of the features class to the implementation
    // in this class. It simply maps NESW coordinates to an index in the dx dy arrays
    private Map<String,Integer> compassMap = Stream.of(
            new AbstractMap.SimpleEntry<>("WW", 0),
            new AbstractMap.SimpleEntry<>("SW", 1),
            new AbstractMap.SimpleEntry<>("SS", 2),
            new AbstractMap.SimpleEntry<>("SE", 3),
            new AbstractMap.SimpleEntry<>("EE", 4),
            new AbstractMap.SimpleEntry<>("NE", 5),
            new AbstractMap.SimpleEntry<>("NN", 6),
            new AbstractMap.SimpleEntry<>("NW", 7))
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

    /**
     * The constructor still uses the dist[] to get the distance of the subgoal to the center of the fire. This however
     * means that the order in which the distances are stored in dist[] determines to which subGoal the distance
     * is assigned. The second constructor uses a HashMap, which gets rid of this problem.
     * @param fireX
     * @param fireY
     * @param dist
     * @param algorithm
     * @param cells
     */
    public OrthogonalSubGoals(int fireX, int fireY, double dist[], String algorithm, List<List<Element>> cells){

        this.fireX = fireX;
        this.fireY = fireY;
        this.algorithm = algorithm;
        this.cells = cells;

        agentGoals = new HashMap<>();

        distMap = new HashMap<>();
        for (String key:compassMap.keySet()){
            distMap.put(key, dist[compassMap.get(key)]);
        }

        subGoals = new HashMap<>();
        for (String key: distMap.keySet()){
            subGoals.put(key, getCorrespondingCell(key));
        }
    }

    /**
     * Using a hashMap instead of an int[] for storing the distances, allows for more robust storage of distance storage.
     * The distance of the subgoal to the center of the fire is not linked to the relative location of that subgoal to
     * the fire. By doing so, the order in which the subGoals are stored in the Map is no longer relevant.
     * @param fireX
     * @param fireY
     * @param distMap
     * @param algorithm
     * @param cells
     */
    public OrthogonalSubGoals(int fireX, int fireY, Map<String,Double> distMap, String algorithm, List<List<Element>> cells){
        this.fireX = fireX;
        this.fireY = fireY;
        this.algorithm = algorithm;
        this.cells = cells;

        agentGoals = new HashMap<>();

        this.distMap = distMap;

        subGoals = new HashMap<>();
        for (String key: distMap.keySet()){
            subGoals.put(key, getCorrespondingCell(key));
        }
    }

    public void selectClosestSubGoal(Agent a){ //TODO: Use deep RL for this step as well
        double minDist= Double.MAX_VALUE;
        String keyNearestGoal = null;
        SubGoal nearestGoal = null;
        for (String key: distMap.keySet()){
            SubGoal temp = new SubGoal(cells, subGoals.get(key), algorithm, a, false);
            if(minDist>temp.getMoveCost()){
                if(!agentGoals.containsValue(key)) {
                    minDist = temp.getMoveCost();
                    keyNearestGoal = key;
                    nearestGoal = temp;
                }
            }
        }
        if (keyNearestGoal==null && nearestGoal == null){
            keyNearestGoal = defaultKey;
            nearestGoal = new SubGoal(cells, subGoals.get(keyNearestGoal), algorithm, a, false);
        }
        if (!agentGoals.keySet().contains(a)){
            agentGoals.put(a, keyNearestGoal);
        } else {
            agentGoals.replace(a, keyNearestGoal);
        }
        updateSubGoal(a, keyNearestGoal, nearestGoal);
    }

    /**
     * give an agent a new subGoal. If it already is on a subgoal, start cutting towards another subgoal. If not on
     * the current subGoal, move towards it.
     * @param agent the agent for which the goals need to be updated.
     */
    public void setNextGoal(Agent agent){
        if (!agentGoals.keySet().contains(agent)){
            agentGoals.put(agent, "WW");
        }
        if (!agentOnGoal(agent)){
            Element goalCell = getCorrespondingCell(agentGoals.get(agent));
            agent.setSubGoal(new SubGoal(cells, goalCell, algorithm, agent, false));
        } else {
            String nextGoal = compassMap
                    .entrySet()
                    .stream()
                    .filter(e -> e.getValue() == ((compassMap.get(agentGoals.get(agent)) + 1) % maxNrGoals))
                    .findAny()
                    .get()
                    .getKey();
            Element goalCell = getCorrespondingCell(nextGoal);
            updateSubGoal(agent, nextGoal, new SubGoal(cells, goalCell, algorithm, agent, true));
        }
    }

    public void updateSubGoal(Agent agent, String key, SubGoal goal){
        agentGoals.replace(agent,key);
        agent.setSubGoal(goal);
    }


    private boolean agentOnGoal(Agent agent){
        return (agent.getX() == xOfGoal(agentGoals.get(agent)) && agent.getY() == yOfGoal(agentGoals.get(agent)));
    }


    private int xOfGoal(String key){
        int goalNr = compassMap.get(key);
        if (dx[goalNr]*dy[goalNr]==0) {
            return Math.toIntExact(Math.round(distMap.get(key)))* dx[goalNr] + fireX;
        } else {
            return Math.toIntExact(Math.round(distMap.get(key)*dx[goalNr]/Math.sqrt(2)))+fireX;
        }
    }

    private int yOfGoal(String key){
        int goalNr = compassMap.get(key);
        if (dx[goalNr]*dy[goalNr]==0) {
            return Math.toIntExact(Math.round(distMap.get(key))) * dy[goalNr] + fireY;
        } else {
            return Math.toIntExact(Math.round(distMap.get(key)*dy[goalNr]/Math.sqrt(2)))+fireY;
        }
    }



    private Element getCorrespondingCell(String key){

        int xDist = xOfGoal(key);
        int yDist = yOfGoal(key);
        if(xDist < 0){
            xDist = 0;
        }
        if(xDist >= cells.size()){
            xDist = cells.size() -1;
        }
        if(yDist < 0){
            yDist = 0;
        }
        if(yDist >= cells.get(0).size()){
            yDist = cells.get(0).size() -1;
        }

        return cells.get(xDist).get(yDist);
    }

    public void updateSubGoal(String key, double dist) {
        Double oldValue = distMap.replace(key, dist);
        if (oldValue == null){
            System.out.println("distMap value not updated successfully!! Selected key : " + key + ", complete keySet: " + Arrays.toString(distMap.keySet().toArray()));
            return;

        }
        subGoals.put(key, getCorrespondingCell(key));
    }

    /**
     * function used to check whether a agent is able to reach the subGoal
     * @param key: The goal which needs to be checked
     * @param a: The agent that is supposed to reach the goal.
     * @return
     */
    public boolean checkSubGoal(String key, Agent a){
        SubGoal goal = new SubGoal(cells, subGoals.get(key), algorithm, a, false);
        System.out.println("Currently looking at subGoal: " + subGoals.get(key).toCoordinates());
        return goal.pathExists();
    }

    /**
     * Debugging function. Removes the goal associated with the key
     * @param key: the to goal to be colored.
     */
    public void paintGoal(String key){
        subGoals.get(key).colorGoal();
    }
}
