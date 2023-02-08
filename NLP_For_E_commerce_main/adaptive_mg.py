import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init

# mostly understood by wjy -----wjy
#========================================Knowing When to Look========================================
class AttentiveCNN( nn.Module ):
    def __init__( self, embed_size, hidden_size ):
        super( AttentiveCNN, self ).__init__()
        
        # ResNet-152 backend
        resnet = models.resnet152( weights=models.ResNet152_Weights.IMAGENET1K_V1)
        modules = list( resnet.children() )[ :-2 ] # delete the last fc layer and avg pool.
        resnet_conv = nn.Sequential( *modules ) # last conv feature
        
        self.resnet_conv = resnet_conv
        self.avgpool = nn.AvgPool2d( 7 )
        self.affine_a = nn.Linear( 2048, hidden_size ) # v_i = W_a * A
        self.affine_b = nn.Linear( 2048, embed_size )  # v_g = W_b * a^g
        
        # Dropout before affine transformation
        self.dropout = nn.Dropout( 0.5 )
        
        self.init_weights()
        
    def init_weights( self ):
        """Initialize the weights."""
        init.kaiming_uniform_( self.affine_a.weight, mode='fan_in' )
        init.kaiming_uniform_( self.affine_b.weight, mode='fan_in' )
        self.affine_a.bias.data.fill_( 0 )
        self.affine_b.bias.data.fill_( 0 )
        
        
    def forward( self, images ):
        '''
        Input: images
        Output: V=[v_1, ..., v_n], v_g
        '''
        
        # Last conv layer feature map
        A = self.resnet_conv( images )
        
        # a^g, average pooling feature map
        a_g = self.avgpool( A )
        a_g = a_g.view( a_g.size(0), -1 )
        
        # V = [ v_1, v_2, ..., v_49 ]
        V = A.view( A.size( 0 ), A.size( 1 ), -1 ).transpose( 1,2 )
        V = F.relu( self.affine_a( self.dropout( V ) ) )
        
        v_g = F.relu( self.affine_b( self.dropout( a_g ) ) )
        
        return V, v_g

# Attention Block for C_hat calculation
class Atten_des( nn.Module ):
    def __init__( self, hidden_size ):
        super( Atten_des, self ).__init__()

        self.affine_v = nn.Linear( hidden_size, 49, bias=False ) # W_v
        self.affine_g = nn.Linear( hidden_size, 49, bias=False ) # W_g
        self.affine_s = nn.Linear( hidden_size, 49, bias=False ) # W_s
        self.affine_h = nn.Linear( 49, 1, bias=False ) # w_h
        
        self.dropout = nn.Dropout( 0.5 )
        self.init_weights()
        
    def init_weights( self ):
        """Initialize the weights."""
        init.xavier_uniform_( self.affine_v.weight )
        init.xavier_uniform_( self.affine_g.weight )
        init.xavier_uniform_( self.affine_h.weight )
        init.xavier_uniform_( self.affine_s.weight )
        
    def forward( self, V, h_t, s_t ):                             # s_t删
        '''
        Input: V=[v_1, v_2, ... v_k], h_t, s_t from LSTM
        Output: c_hat_t, attention feature map
        '''
        
        # W_v * V + W_g * h_t * 1^T
        content_v = self.affine_v( self.dropout( V ) ).unsqueeze( 1 ) \
                    + self.affine_g( self.dropout( h_t ) ).unsqueeze( 2 )
        
        # z_t = W_h * tanh( content_v )
        z_t = self.affine_h( self.dropout( torch.tanh( content_v ) ) ).squeeze( 3 )
        alpha_t = F.softmax( z_t.view( -1, z_t.size( 2 ) ), dim = -1).view( z_t.size( 0 ), z_t.size( 1 ), -1 )
        
        # Construct c_t: B x seq x hidden_size
        c_t = torch.bmm( alpha_t, V ).squeeze( 2 )
        
        # W_s * s_t + W_g * h_t
        content_s = self.affine_s( self.dropout( s_t ) ) + self.affine_g( self.dropout( h_t ) )
        # w_t * tanh( content_s )
        z_t_extended = self.affine_h( self.dropout( torch.tanh( content_s ) ) )
        
        # Attention score between sentinel and image content
        extended = torch.cat( ( z_t, z_t_extended ), dim=2 )
        alpha_hat_t = F.softmax( extended.view( -1, extended.size( 2 ) ), dim = -1 ).view( extended.size( 0 ), extended.size( 1 ), -1 )
        beta_t = alpha_hat_t[ :, :, -1 ]
        
        # c_hat_t = beta * s_t + ( 1 - beta ) * c_t
        beta_t = beta_t.unsqueeze( 2 )
        c_hat_t = beta_t * s_t + ( 1 - beta_t ) * c_t

        return c_hat_t, alpha_t, beta_t

# Attention Block for C_hat calculation
class Atten_tit( nn.Module ):
    def __init__( self, hidden_size ):
        super( Atten_tit, self ).__init__()

        self.affine_v = nn.Linear( hidden_size, 49, bias=False ) # W_v
        self.affine_g = nn.Linear( hidden_size, 49, bias=False ) # W_g
        self.affine_s = nn.Linear( hidden_size, 49, bias=False ) # W_s
        self.affine_h = nn.Linear( 49, 1, bias=False ) # w_h
        
        self.dropout = nn.Dropout( 0.5 )
        self.init_weights()
        
    def init_weights( self ):
        """Initialize the weights."""
        init.xavier_uniform_( self.affine_v.weight )
        init.xavier_uniform_( self.affine_g.weight )
        init.xavier_uniform_( self.affine_h.weight )
        init.xavier_uniform_( self.affine_s.weight )
        
    def forward( self, hiddens_des, hiddens_title ):                             # s_t删
        #c_-t_-1 = (V, h_t_1)
        #c_t_2 = (V, h_t_2)
        #c_t = (V, h_t_2)) +(h_t_1, h_t_2)
        
        '''
        Input: V=[v_1, v_2, ... v_k], h_t, s_t from LSTM
        Output: c_hat_t, attention feature map
        '''
        
        # W_v * V + W_g * h_t * 1^T
        content_v = self.affine_v( self.dropout( hiddens_des) ).unsqueeze( 1 ) \
                    + self.affine_g( self.dropout( hiddens_title ) ).unsqueeze( 2 )
        
        # z_t = W_h * tanh( content_v )
        z_t = self.affine_h( self.dropout( torch.tanh( content_v ) ) ).squeeze( 3 )
        alpha_t = F.softmax( z_t.view( -1, z_t.size( 2 ) ), dim = -1).view( z_t.size( 0 ), z_t.size( 1 ), -1 )
        
        # Construct c_t: B x seq x hidden_size
        c_t = torch.bmm( alpha_t, hiddens_des ).squeeze( 2 )
        
        # W_s * s_t + W_g * h_t
        # content_s = self.affine_s( self.dropout( s_t ) ) + self.affine_g( self.dropout( h_t ) )
        # w_t * tanh( content_s )
        # z_t_extended = self.affine_h( self.dropout( torch.tanh( content_s ) ) )
        
        # Attention score between sentinel and image content
        # extended = torch.cat( ( z_t, z_t_extended ), dim=2 )
        # alpha_hat_t = F.softmax( extended.view( -1, extended.size( 2 ) ), dim = -1 ).view( extended.size( 0 ), extended.size( 1 ), -1 )
        # beta_t = alpha_hat_t[ :, :, -1 ]
        
        # c_hat_t = beta * s_t + ( 1 - beta ) * c_t
        # beta_t = beta_t.unsqueeze( 2 )
        c_hat_t = c_t

        return c_hat_t, alpha_t

# Sentinel BLock    
class Sentinel( nn.Module ):
    def __init__( self, input_size, hidden_size ):
        super( Sentinel, self ).__init__()

        self.affine_x = nn.Linear( input_size, hidden_size, bias=False )
        self.affine_h = nn.Linear( hidden_size, hidden_size, bias=False )
        
        # Dropout applied before affine transformation
        self.dropout = nn.Dropout( 0.5 )
        
        self.init_weights()
        
    def init_weights( self ):
        init.xavier_uniform_( self.affine_x.weight )
        init.xavier_uniform_( self.affine_h.weight )
        
    def forward( self, x_t, h_t_1, cell_t ):
        
        # g_t = sigmoid( W_x * x_t + W_h * h_(t-1) )        
        gate_t = self.affine_x( self.dropout( x_t ) ) + self.affine_h( self.dropout( h_t_1 ) )
        gate_t = torch.sigmoid( gate_t )
        
        # Sentinel embedding
        s_t =  gate_t * torch.tanh( cell_t )
        
        return s_t

# Adaptive Attention Block: C_t, Spatial Attention Weights, Sentinel embedding    
class AdaptiveBlock( nn.Module ):
    
    def __init__( self, embed_size, hidden_size, vocab_size ):
        super( AdaptiveBlock, self ).__init__()

        # Sentinel block
        self.sentinel = Sentinel( embed_size * 2, hidden_size )
        
        # Image Spatial Attention Block
        self.atten_des = Atten_des( hidden_size )
        self.atten_tit = Atten_tit( hidden_size )
        
        # Final Caption generator
        self.mlp = nn.Linear( hidden_size, vocab_size )
        
        # Dropout layer inside Affine Transformation
        self.dropout = nn.Dropout( 0.5 )
        
        self.hidden_size = hidden_size
        self.init_weights()
        
    def init_weights( self ):
        '''
        Initialize final classifier weights
        '''
        init.kaiming_normal_( self.mlp.weight, mode='fan_in' )
        self.mlp.bias.data.fill_( 0 )
        
        
    def forward( self, x_des, x_tit, hiddens_des, hiddens_title, cells_des, cells_title, V ):
        
        ##DESCRIPTION
        # hidden for sentinel should be h0-ht-1
        h0_des = self.init_hidden( x_des.size(0) )[0].transpose( 0,1 )
        
        # h_(t-1): B x seq x hidden_size ( 0 - t-1 )
        if hiddens_des.size( 1 ) > 1:
            hiddens_t_1_des = torch.cat( ( h0_des, hiddens_des[ :, :-1, : ] ), dim=1 )
        else:
            hiddens_t_1_des = h0_des

        # Get Sentinel embedding, it's calculated blockly    
        sentinel_des = self.sentinel( x_des, hiddens_t_1_des, cells_des )
        
        # Get C_t, Spatial attention, sentinel score
        c_hat_des, atten_weights, beta = self.atten_des( V, hiddens_des, sentinel_des )
        scores_des = self.mlp( self.dropout( c_hat_des + hiddens_des ) )


        ##TITLE
        h0_tit = self.init_hidden( x_tit.size(0) )[0].transpose( 0,1 )
        
        # h_(t-1): B x seq x hidden_size ( 0 - t-1 )
        if hiddens_title.size( 1 ) > 1:
            hiddens_t_1_tit = torch.cat( ( h0_tit, hiddens_title[ :, :-1, : ] ), dim=1 )
        else:
            hiddens_t_1_tit = h0_tit

        # Get Sentinel embedding, it's calculated blockly    
        sentinel_tit = self.sentinel( x_tit, hiddens_t_1_tit, cells_title )

        c_hat_title, atten_weights, beta = self.atten_des( V, hiddens_title, sentinel_tit )

        c_hat_title_to_des, atten_weights = self.atten_tit( hiddens_des, hiddens_title) 
        # xianshi
        c_hat_title = c_hat_title + c_hat_title_to_des
        scores_title = self.mlp( self.dropout( c_hat_title + hiddens_title ) )
        # Final score along vocabulary
        
        return scores_des, scores_title# , atten_weights# , beta
    
    def init_hidden( self, bsz ):
        '''
        Hidden_0 & Cell_0 initialization
        '''
        weight = next( self.parameters() ).data
        
        if torch.cuda.is_available():
            return ( Variable( weight.new( 1 , bsz, self.hidden_size ).zero_().cuda() ),
                    Variable( weight.new( 1,  bsz, self.hidden_size ).zero_().cuda() ) ) 
        else: 
            return ( Variable( weight.new( 1 , bsz, self.hidden_size ).zero_() ),
                    Variable( weight.new( 1,  bsz, self.hidden_size ).zero_() ) ) 
    

# Caption Decoder
class Decoder( nn.Module ):
    def __init__( self, embed_size, vocab_size, hidden_size ):
        super( Decoder, self ).__init__()

        # word embedding
        self.embed = nn.Embedding( vocab_size, embed_size )
        
        # LSTM decoder: input = [ w_t; v_g ] => 2 x word_embed_size;
        self.LSTM1 = nn.LSTM( embed_size * 2, hidden_size, 1, batch_first=True )
        self.LSTM2 = nn.LSTM( embed_size * 2, hidden_size, 1, batch_first=True )
        
        # Save hidden_size for hidden and cell variable 
        self.hidden_size = hidden_size
        
        # Adaptive Attention Block: Sentinel + C_hat + Final scores for caption sampling
        self.adaptive = AdaptiveBlock( embed_size, hidden_size, vocab_size )
        
    def forward( self, V, v_g , description, captions, states=None ):
        
        # Word Embedding
        embeddings_des = self.embed( description )
        
        # x_t = [w_t;v_g]
        x_des = torch.cat( ( embeddings_des, v_g.unsqueeze( 1 ).expand_as( embeddings_des ) ), dim=2 )
        # print(x.size()) [20,15,512]
        # Hiddens: Batch x seq_len x hidden_size
        # Cells: seq_len x Batch x hidden_size, default setup by Pytorch
        if torch.cuda.is_available():
            hiddens_des = Variable( torch.zeros( x_des.size(0), x_des.size(1), self.hidden_size ).cuda() )
            cells_des = Variable( torch.zeros( x_des.size(1), x_des.size(0), self.hidden_size ).cuda() )
        else:
            hiddens_des = Variable( torch.zeros( x_des.size(0), x_des.size(1), self.hidden_size ) )
            cells_des = Variable( torch.zeros( x_des.size(1), x_des.size(0), self.hidden_size ) )            
        
        # Recurrent Block
        # Retrieve hidden & cell for Sentinel simulation
        for time_step in range( x_des.size( 1 ) ):
            
            # Feed in x_t one at a time
            x_t = x_des[ :, time_step, : ]
            x_t = x_t.unsqueeze( 1 )
            
            h_t, states = self.LSTM1( x_t, states )
            # Save hidden and cell
            hiddens_des[ :, time_step, : ] = h_t.squeeze( 1 )  # Batch_first
            cells_des[ time_step, :, : ] = states[ 1 ]
        
        # cell: Batch x seq_len x hidden_size
        cells_des = cells_des.transpose( 0, 1 )

    

        # Title Embedding
        embeddings_tit = self.embed( captions )
        
        # x_t = [w_t;v_g]
        x_tit = torch.cat( ( embeddings_tit, v_g.unsqueeze( 1 ).expand_as( embeddings_tit ) ), dim=2 )
        # print(x.size()) [20,15,512]
        # Hiddens: Batch x seq_len x hidden_size
        # Cells: seq_len x Batch x hidden_size, default setup by Pytorch
        if torch.cuda.is_available():
            hiddens_tit = Variable( torch.zeros( x_tit.size(0), x_tit.size(1), self.hidden_size ).cuda() )
            cells_tit = Variable( torch.zeros( x_tit.size(1), x_tit.size(0), self.hidden_size ).cuda() )
        else:
            hiddens_tit = Variable( torch.zeros( x_tit.size(0), x_tit.size(1), self.hidden_size ) )
            cells_tit = Variable( torch.zeros( x_tit.size(1), x_tit.size(0), self.hidden_size ) )            
        
        # Recurrent Block
        # Retrieve hidden & cell for Sentinel simulation
        for time_step in range( x_tit.size( 1 ) ):
            
            # Feed in x_t one at a time
            x_t = x_tit[ :, time_step, : ]
            x_t = x_t.unsqueeze( 1 )
            
            h_t, states = self.LSTM2( x_t, states )
            # Save hidden and cell
            hiddens_tit[ :, time_step, : ] = h_t.squeeze( 1 )  # Batch_first
            cells_tit[ time_step, :, : ] = states[ 1 ]
        
        # cell: Batch x seq_len x hidden_size
        cells_tit = cells_tit.transpose( 0, 1 )

        # Data parallelism for adaptive attention block
        if torch.cuda.device_count() > 1:
            device_ids = range( torch.cuda.device_count() )
            adaptive_block_parallel = nn.DataParallel( self.adaptive, device_ids=device_ids )
            
            scores_des, scores_title = adaptive_block_parallel( x_des, x_tit, hiddens_des, hiddens_tit, cells_des, cells_tit, V )
        else:
            scores_des, scores_title = self.adaptive( x_des, x_tit, hiddens_des, hiddens_tit, cells_des, cells_tit, V )
        
      
        # Return states for Caption Sampling purpose
        return scores_des, scores_title
    
        

# Whole Architecture with Image Encoder and Caption decoder        
class Encoder2Decoder( nn.Module ):
    def __init__( self, embed_size, vocab_size, hidden_size ):
        super( Encoder2Decoder, self ).__init__()
        
        # Image CNN encoder and Adaptive Attention Decoder
        self.encoder = AttentiveCNN( embed_size, hidden_size )
        self.decoder = Decoder( embed_size, vocab_size, hidden_size )
        
        
    def forward( self, images, description, captions, lengths_des, lengths_cap ):
        
        # Data parallelism for V v_g encoder if multiple GPUs are available
        # V=[ v_1, ..., v_k ], v_g in the original paper
        if torch.cuda.device_count() > 1:
            device_ids = range( torch.cuda.device_count() )
            encoder_parallel = torch.nn.DataParallel( self.encoder, device_ids=device_ids )
            V, v_g = encoder_parallel( images ) 
        else:
            V, v_g = self.encoder( images )
        
        # Language Modeling on word prediction
        scores_des, scores_title = self.decoder( V, v_g, description, captions )
        
        # Pack it to make criterion calculation more efficient
        packed_scores_des = pack_padded_sequence( scores_des, lengths_des, batch_first=True )
        packed_scores_title = pack_padded_sequence( scores_title, lengths_cap, batch_first=True,enforce_sorted=False )
        
        return packed_scores_des, packed_scores_title
    
    # Caption generator
    def sampler( self, images, max_len=20 ):
        """
        Samples captions for given image features (Greedy search).
        """
        
        # Data parallelism if multiple GPUs
        if torch.cuda.device_count() > 1:
            device_ids = range( torch.cuda.device_count() )
            encoder_parallel = torch.nn.DataParallel( self.encoder, device_ids=device_ids )
            V, v_g = encoder_parallel( images ) 
        else:    
            V, v_g = self.encoder( images )
            
        # Build the starting token Variable <start> (index 1): B x 1
        if torch.cuda.is_available():
            captions = Variable( torch.LongTensor( images.size( 0 ), 1 ).fill_( 1 ).cuda() )
            description = Variable( torch.LongTensor( images.size( 0 ), 1 ).fill_( 1 ).cuda() )
        else:
            captions = Variable( torch.LongTensor( images.size( 0 ), 1 ).fill_( 1 ) )
            description = Variable( torch.LongTensor( images.size( 0 ), 1 ).fill_( 1 ) )
        
        # Get generated caption idx list, attention weights and sentinel score
        sampled_ids = []
        # attention = []
        # Beta = []
        
        # Initial hidden states
        states = None

        for i in range( max_len ):

            scores_des, scores_title = self.decoder( V, v_g, description, captions, states ) 
            predicted = scores_title.max( 2 )[ 1 ] # argmax
            # print(predicted)
            captions = predicted
            # print(captions)
            
            # Save sampled word, attention map and sentinel at each timestep
            sampled_ids.append( captions )
            # attention.append( atten_weights )
            # Beta.append( beta )
        # print(sampled_ids)
        # caption: B x max_len
        # attention: B x max_len x 49
        # sentinel: B x max_len
        sampled_ids = torch.cat( sampled_ids, dim=1 )
        # attention = torch.cat( attention, dim=1 )
        # Beta = torch.cat( Beta, dim=1 )
        # print(sampled_ids)
        return sampled_ids# , attention, Beta
