# MCTS その３　#
---

[//]: # (Image References)
[plot1]: ./ai_first1.png


今回久々の更新になります。MCTSのその３として今回も記事を書かせていただきます。
しばらく時間があいてしまいましたので、内容的には乏しいものとなっているかもしれません。
その１、その２と数百のアクセスがありましたが、記事で紹介しました理論についての誤った指摘などが未だ報告されておりませんので、胸をなで下ろしている次第です。

前回に引き続きMCTSのProgramming Logicについて具体的に書きます。今回は実際のCodingについて詳細に書いていきます。

---
## Program構成について

Programは次のようなPartに分かれます。

### 1. GameStateについて

ゲームを行うボードを管理するPython Classです。Action、Stateなど重要な要素を含むmethodが多数含まれていますが、今回は動作の確認とA Survey of MCTSのメインロジックとリンクさせたいので、実装自体非常に軽く作成してあります。
Monte Carlo側が⭕️として、対戦相手(人間の手)を❌としてあります。
今回は❌先行で、Monte Carlo側がどのようにが自律的に学習を進めてよい打ち手を学習していくかをシミュレーションするようにしてあります。
Alpha Goにおいてもプログラムが進化して新たな手を自ずから考案していくことでより強力な打ち手となっていきます。

* 初期のConstructorは下記のように非常に簡単です。
全部で9ターンで勝負が付きますので、保有するnumpy arrayも9個あれば必要なのですが、numpyのrange関数の初期値0は使用しないので10個用意してあります。つまりrangeの引数は1から使用します。

```
class GameState(object):
    """
    Represents a Tic Tac Toe game.
    The state consists of a 3x3 game board with each position occupied by:
      ' ' (empty square)
      'X' (X mark)
      'O' (O mark)
    as well as the following terminal states:
      X won
      O won
      Tie
    """
    def __init__(self):
        # Begin with an empty game board

        self.board = np.array([" "] * 10)
        self.NUM_MOVES = 9
```

* GameStateでの重要なルーチンといえば、winner functionになります。今回は９マスの三目並べですので、勝敗ルーチンをすべてハードコード下記のようにしてあります。もし、大きなマスで行う場合は、より効率的かつScalabilityに対応できるロジックが求められるところですが、今回は自分の趣味の範囲ですのでマニュアルでハードコードしました。

```
def winner(self):
    """
    Checks if the game state is a terminal state.
    :return: If it is not, returns None; if it is, returns 'X' or 'O'
    indicating who is the winner; if it is a tie, returns 'Tie'
    """
    for player in ['X', 'O']:
        # Check for winning vertical lines
        for cols in [ [1,4,7], [2,5,8], [3,6,9]      ]:
            accum = 0
            for col in cols:
                if self.board[col] == player:
                    accum += 1
            if accum == 3:
                return player

        # Check for winning horizontal lines
        for rows in [ [1,2,3], [4,5,6], [7,8,9] ] :
            accum = 0
            for row in rows:
                if self.board[row] == player:
                    accum += 1
            if accum == 3:
                return player

        # Check for winning diagonal lines (there are 2 possibilities)
        option1 = [self.board[1],
                   self.board[5],
                   self.board[9]]
        option2 = [self.board[3],
                   self.board[5],
                   self.board[7]]
        if all(marker == player for marker in option1) \
                or all(marker == player for marker in option2):
            return player

    # Check for ties, defined as a board arrangement in which there are no
    # open board positions left and there are no winners (note that the
    # tie is not being detected ahead of time, as could potentially be
    # done)
    accum = 0
    for row in range(10)[1:]:
        if self.board[row] == ' ':
            accum += 1
    if accum == 0:
        return 'Tie'

    return None

```


### 2. mcts_player について

mtcs_playerでは、MTCSpolicyとRandomPolicy Classが記述されます。RandomPolicyは、人間の手と考えていますが、実験のため可能なマスがあれば、とにかくXをつけるようにしてあります。


#### 2.1 RandomPolicyについて

下記のように非常に簡単なクラスで構成されます。
GemeStateのlegal_moveで可能なものを適当にActionとして定義しているだけです。

```
class RandomPolicy(Policy):
    def move(self, state):
        """Chooses moves randomly from the legal moves in a given state"""
        legal_moves = state.legal_moves()
        #print("<RandomPolicy> legal_moves : %s" % legal_moves)
        idx = np.random.randint(len(legal_moves))
        return legal_moves[idx]
```


#### 2.2 MCTSPolicyについて

MCTSPolicyのConstructorは下記のようになります。MCTSPolicy内部で保有するTree情報の格納はnetworkxのDiGraphを使用しました。Node Classを作成してシミュレーションを行いまいしたが、結果が予想とずれてしまったたため、あるところを参考にしてDiGraphに作り替えました。


```
class MCTSPolicy(Policy):

    def __init__(self, player="X"):
        """
        Implementation of Monte Carlo Tree Search
        Creates a root of an MCTS tree to keep track of the information
        obtained throughout the course of the game in the form of a tree
        of MCTS nodes
        The data structure of a node consists of:
          - the game state which it corresponds to
          - w, the number of wins that have occurred at or below it in the tree
          - n, the number of plays that have occurred at or below it in the tree
          - expanded, whether all the children (legal moves) of the node have
            been added to the tree
        To access the node attributes, use the following format. For example,
        to access the attribute 'n' of the root node:
          policy = MCTSPolicy()
          current_node = policy.root
          policy.tree.node[current_node]['n']
        """
        self.digraph = nx.DiGraph()
        self.player = player
        self.num_simulations = 0
        # Constant parameter to weight exploration vs. exploitation for UCT
        self.uct_c = 1. / np.sqrt(2)

        self.node_counter = 0

        empty_board = GameState()
        self.digraph.add_node(self.node_counter,w=0,vn=0,uct=0, expanded=False, state=empty_board)
        empty_board_node_id = self.node_counter
        self.node_counter += 1

        self.last_move = None

```

MCTSPolicyでは、下記snippetに示されているように、メインルーチンがuctsearchになります。
ここでは、Treepolicy、 Defaultpolicy、BackupルーチンをそれぞれCallしています。
for loopで作成したシミュレーションは、100回行い、Treepolicy、Defaultpolicyそしてbackuoの処理を行います。この辺の処理は前回説明した同名のプロセスをさしているので、それぞれ参照することができます。

```
def uctsearch(self,starting_state):

    starting_node = None
    starting_state = copy.deepcopy(starting_state)

    found = False
    for n in self.digraph.nodes():
        if self.digraph.node[n]["state"] == starting_state:
            mylogger.debug("matched existing node id -> %d ", n)
            starting_node = n
            found = True
    if not found:
        self.digraph.add_node(self.node_counter, w=0,vn=0,uct=0, expanded=False, state = starting_state)
        mylogger.debug("root is not in digraph. add node. %d " , self.node_counter)
        starting_node = self.node_counter
        self.node_counter += 1

    for i in range(100):
        node = self.treepolicy(starting_node)
        reward = self.defaultpolicy(node)

        #mylogger.debug("reward", reward)
        self.backup(node,reward)

    best_child_id, move = self.best(starting_node)

    mylogger.debug("best Child id --> %d",best_child_id)
    mylogger.debug("best move --> %d", move)
    return move

```

### 3. 実行について



```
python gameplay.py
```
で実行するとログが表示されながら、全対戦の記録をpythonのリストに保存していきます。
最後に対戦勝率の図が下記のようにplotされます。  
200回ほどの対戦を記録したものですが、75回目より勝率が上昇しはじめ、図のbottomで0.24の勝率か
ら0.38の勝率までたどり着き

![alt text][plot1]
