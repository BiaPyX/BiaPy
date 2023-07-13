import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from skimage import data
from skimage.transform import resize
import matplotlib.pyplot as plt

from .tr_layers import Patches, PatchEncoder, TransformerBlock

class MaskedAutoencoder(Model):
    def __init__(self, model_input_shape, patch_size, enc_hidden_size, enc_transformer_layers, enc_num_heads, enc_mlp_head_units, 
        mask_proportion=0.75, enc_dropout=0.1, dec_hidden_size=64, dec_transformer_layers=2, dec_num_heads=4, dec_mlp_head_units=128, 
        dec_dropout=0.1):
        """
        MAE architecture. `MAE paper <https://arxiv.org/abs/2111.06377>`__.

        Parameters
        ----------
        model_input_shape : 3D/4D tuple
            Dimensions of the input image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.
            
        patch_size : int
            Size of the patches that are extracted from the input image. As an example, to use ``16x16`` 
            patches, set ``patch_size = 16``.

        enc_hidden_size : int
            Dimension of the embedding space for the enconder.

        enc_transformer_layers : int
            Number of transformer encoder layers for the enconder.

        enc_num_heads : int
            Number of heads in the multi-head attention layer for the enconder.

        enc_mlp_head_units : int
            Size of the dense layer of the final classifier for the enconder. 

        mask_proportion : float, optional
            Propotion of input image to be masked. 

        enc_dropout : bool, optional
            Dropout rate for the encoder (can be a list of dropout rates for each layer).

        dec_hidden_size : bool, optional
            Dimension of the embedding space for the decoder.

        dec_transformer_layers : int, optional
            Number of transformer encoder layers for the decoder. 

        dec_num_heads : bool, optional
            Number of heads in the multi-head attention layer for the decoder.

        dec_mlp_head_units : bool, optional
            Size of the dense layer of the final classifier for the decoder. 

        dec_dropout : int, optional
            Dropout rate for the decoder (can be a list of dropout rates for each layer). 
        """
        
        super().__init__()
        self.model_input_shape = model_input_shape
        self.patch_size = patch_size
        self.enc_hidden_size = enc_hidden_size
        self.enc_transformer_layers = enc_transformer_layers
        self.enc_num_heads = enc_num_heads
        self.enc_mlp_head_units = enc_mlp_head_units
        self.mask_proportion= mask_proportion
        self.enc_dropout=enc_dropout
        self.dec_hidden_size=dec_hidden_size
        self.dec_transformer_layers=dec_transformer_layers
        self.dec_num_heads=dec_num_heads
        self.dec_mlp_head_units=dec_mlp_head_units
        self.dec_dropout=dec_dropout

        self.channels = self.model_input_shape[-1]
        if len(self.model_input_shape) == 4:
            self.dims = 3 
            self.patch_dims = patch_size*patch_size*patch_size*model_input_shape[-1]
        else:
            self.dims = 2
            self.patch_dims = patch_size*patch_size*model_input_shape[-1]
        self.num_patches = (self.model_input_shape[0]//patch_size)**self.dims

        # Patch creation 
        # 2D: (B, num_patches^2, patch_dims)
        # 3D: (B, num_patches^3, patch_dims)
        self.patch_layer = Patches(self.patch_size, self.patch_dims, self.dims)

        # Patch encoder
        # 2D: (B, num_patches^2, hidden_size)
        # 3D: (B, num_patches^3, hidden_size)
        self.patch_encoder = PatchEncoder(num_patches=self.num_patches, 
            hidden_size=enc_hidden_size, patch_dims=self.patch_dims, mask_proportion=mask_proportion)
        # y = PatchEncoder(num_patches=num_patches, hidden_size=hidden_size)(y)

        self.encoder = self.__create_encoder()
        self.decoder = self.__create_decoder()
    
    def __create_encoder(self):
        """Creates an enconder for MAE. The structure is basically a slightly modified ViT. """ 
        inputs = Input((None, self.enc_hidden_size))
        y = inputs

        for i in range(self.enc_transformer_layers):
            y, _ = TransformerBlock(
                num_heads=self.enc_num_heads,
                mlp_dim=self.enc_mlp_head_units,
                dropout=self.enc_dropout,
                name=f"Transformer/encoderblock_{i}",
            )(y)

        outputs = layers.LayerNormalization(epsilon=1e-6)(y)
        return Model(inputs, outputs, name="mae_encoder")

    def __create_decoder(self):
        """Creates an small decoder for MAE. The structure is basically a slightly modified ViT. """ 
        inputs = Input((self.num_patches, self.enc_hidden_size))
        y = layers.Dense(self.dec_hidden_size)(inputs)

        for i in range(self.dec_transformer_layers):
            y, _ = TransformerBlock(
                num_heads=self.dec_num_heads,
                mlp_dim=self.dec_mlp_head_units,
                dropout=self.dec_dropout,
                name=f"Transformer/decoderblock_{i}",
            )(y)

        y = layers.LayerNormalization(epsilon=1e-6)(y)
        y = layers.Flatten()(y)
        pre_final = layers.Dense(units=np.prod(self.model_input_shape), activation="sigmoid")(y)
        outputs = layers.Reshape(self.model_input_shape)(pre_final)

        return Model(inputs, outputs, name="mae_decoder")

    def calculate_loss(self, images, test=False):
        # Patch images.
        patches = self.patch_layer(images)

        # Encode the patches.
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.patch_encoder(patches)

        # Pass the unmasked patch to the encoder.
        encoder_outputs = self.encoder(unmasked_embeddings)

        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = tf.concat([encoder_outputs, masked_embeddings], axis=1)

        # Decode the inputs.
        decoder_outputs = self.decoder(decoder_inputs)
        decoder_patches = self.patch_layer(decoder_outputs)

        loss_patch = tf.gather(patches, mask_indices, axis=1, batch_dims=1)
        loss_output = tf.gather(decoder_patches, mask_indices, axis=1, batch_dims=1)

        # Compute the total loss.
        total_loss = self.compiled_loss(loss_patch, loss_output)

        return total_loss, loss_patch, loss_output

    def train_step(self, images):
        with tf.GradientTape() as tape:
            total_loss, loss_patch, loss_output = self.calculate_loss(images)

        # Apply gradients.
        train_vars = [
            self.patch_layer.trainable_variables,
            self.patch_encoder.trainable_variables,
            self.encoder.trainable_variables,
            self.decoder.trainable_variables,
        ]
        grads = tape.gradient(total_loss, train_vars)
        tv_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)

        # Report progress.
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        # imgs, _ = images
        total_loss, loss_patch, loss_output = self.calculate_loss(images, test=True)

        # Update the trackers.
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x):
        x = self.patch_layer(x)
        # Encode the patches.
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            _,
            _,
        ) = self.patch_encoder(x)

        # Pass the unmasked patch to the encoder.
        encoder_outputs = self.encoder(unmasked_embeddings)

        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = tf.concat([encoder_outputs, masked_embeddings], axis=1)

        # Decode the inputs.
        decoder_outputs = self.decoder(decoder_inputs)
        return decoder_outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'model_input_shape': self.model_input_shape, 
            'patch_size': self.patch_size,
            'enc_hidden_size': self.enc_hidden_size, 
            'enc_transformer_layers': self.enc_transformer_layers, 
            'enc_num_heads': self.enc_num_heads,
            'enc_mlp_head_units': self.enc_mlp_head_units,
            'mask_proportion': self.mask_proportion,
            'enc_dropout': self.enc_dropout,
            'dec_hidden_size': self.dec_hidden_size,
            'dec_transformer_layers': self.dec_transformer_layers,
            'dec_num_heads': self.dec_num_heads,
            'dec_mlp_head_units': self.dec_mlp_head_units,
            'dec_dropout': self.dec_dropout,
            'channels': self.channels,
            'dims': self.dims,
            'patch_dims': self.patch_dims,
            'num_patches': self.num_patches,
            'patch_layer': self.patch_layer,
            'patch_encoder': self.patch_encoder,
            'encoder': self.encoder,
            'decoder': self.decoder,
        })
        return config

    def create_image_example(self, image=None):  
        # Toy image
        if image is None:
            image = data.astronaut() 
        image = resize(image, (self.model_input_shape[0], model_input_shape[1]), anti_aliasing=True)
        image = np.expand_dims(np.expand_dims(np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]),0),-1)
        image = np.tile(image, model_input_shape[-1])
        patches = self.patch_layer(images=image)
        
        # only toy example, so it's 0 but these two steps are written so we can use for generic images
        random_index = np.random.choice(patches.shape[0]) 
        gray = True if image[random_index].shape[-1] == 1 else False

        # Create figure grid
        fig = plt.figure(figsize=(24, 4))
        n = int(np.sqrt(patches.shape[1]))
        grid = plt.GridSpec(n, n*4, wspace = .25, hspace = .25)

        # Original image
        ax = fig.add_subplot(grid[:,:n])
        ax.set_axis_off()
        ax.set_title("Original image")
        if gray:
            plt.imshow(tf.keras.utils.array_to_img(image[random_index]), cmap='gray')
        else:
            plt.imshow(tf.keras.utils.array_to_img(image[random_index]))

        # Patches (created using patch_layer)
        ax = fig.add_subplot(grid[:,n:n*2])
        ax.axis('off')
        ax.set_title('Patches')
        for i, patch in enumerate(patches[random_index]):
            ax = fig.add_subplot(grid[i//n , n+(i%n)])
            ax.set_axis_off()
            patch_img = tf.reshape(patch, patch_dims)
            if image.shape[-1] == 1:
                plt.imshow(tf.keras.utils.img_to_array(patch_img), cmap='gray')
            else:
                plt.imshow(tf.keras.utils.img_to_array(patch_img))
            
        # Chose the same chose image and try reconstructing the patches into the original image.
        rimage = self.patch_layer.reconstruct_from_patch(patches[random_index])
        ax = fig.add_subplot(grid[:,n*2:(n*2)+n])
        ax.set_axis_off()
        ax.set_title("Reconstructed image")
        if gray:
            plt.imshow(tf.keras.utils.array_to_img(rimage), cmap='gray')
        else:
            plt.imshow(tf.keras.utils.array_to_img(rimage))

        # Get the embeddings and positions.
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.patch_encoder(patches=patches)
        new_patch, random_index = self.patch_encoder.generate_masked_image(patches, unmask_indices)

        # Masked image
        ax = fig.add_subplot(grid[:,n*3:(n*3)+n])
        ax.set_axis_off()
        ax.set_title("Masked image")
        masked_img = self.patch_layer.reconstruct_from_patch(new_patch)
        if gray:
            plt.imshow(tf.keras.utils.array_to_img(masked_img), cmap='gray')
        else:
            plt.imshow(tf.keras.utils.array_to_img(masked_img))

        # Save the final image
        plt.savefig('astronaut_mae.png')

def MAE(input_shape, create_mae_example=False, **args):
    """ Returns MAE model. """
    mae_model = MaskedAutoencoder(model_input_shape=input_shape, **args)

    if create_mae_example:
        mae_model.create_image_example()

    # Build model with 1 of batch size so we can do model.summary() after
    mae_model.build((1,) + input_shape)

    return mae_model