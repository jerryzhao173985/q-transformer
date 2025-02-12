648|            for grad_accum_step in range(self.grad_accum_every):
649|                is_last = grad_accum_step == (self.grad_accum_every - 1)
650|                context = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext
651|                # --- modified autocast context manager ---
652|                device_type = self.accelerator.device.type
653|                if device_type == 'mps':
654|                    autocast_context = torch.autocast(device_type='mps', dtype=torch.bfloat16)
655|                else:
656|                    autocast_context = self.accelerator.autocast()
657|
658|                with autocast_context, context():
659|                    loss, (td_loss, conservative_reg_loss) = self.learn(
660|                        *next(replay_buffer_iter),
661|                        min_reward = min_reward,
662|                        monte_carlo_return = monte_carlo_return
663|                    )
664|                    self.accelerator.backward(loss / self.grad_accum_every) 