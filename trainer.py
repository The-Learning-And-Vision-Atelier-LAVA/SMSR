import os
import utility
import torch
from decimal import Decimal


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        # lr schedule
        lr = 2e-4 * (2 ** -((epoch) // 200))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            _, _, outH, outW = hr.size()

            # update tau for gumbel softmax
            tau = max(1 - (epoch - 1) / 500, 0.4)
            for m in self.model.modules():
                if hasattr(m, '_set_tau'):
                    m._set_tau(tau)

            # inference
            self.optimizer.zero_grad()
            sr, sparsity = self.model(lr, idx_scale)

            # losses
            loss_SR = self.loss(sr, hr)
            loss_sparsity = sparsity.mean()
            lambda0 = 0.1
            lambda_sparsity = min((epoch - 1) / 50, 1) * lambda0
            loss = loss_SR + lambda_sparsity * loss_sparsity

            # backpropagation
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t[Sparsity:{:.3f}]\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    float(sparsity.round().mean().data.cpu()),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        target = self.model
        torch.save(
            target.state_dict(),
            os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
        )

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()

        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.loader_test.dataset.set_scale(idx_scale)
                eval_acc = 0
                eval_acc_ssim = 0

                # Kernel split
                # Note that, this part of code does not need to be executed at each run.
                # After training, one can run this part of code once and save the splitted kernels.
                for m in self.model.modules():
                    if hasattr(m, '_prepare'):
                        m._prepare()

                for idx_img, (lr, hr, filename, _) in enumerate(self.loader_test):
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)
                    lr, hr = self.crop_border(lr, hr, scale)

                    sr = self.model(lr, idx_scale)

                # run a second time to record inference time
                for idx_img, (lr, hr, filename, _) in enumerate(self.loader_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)
                    lr, hr = self.crop_border(lr, hr, scale)

                    timer_test.tic()
                    sr = self.model(lr, idx_scale)
                    timer_test.hold()

                    sr = utility.quantize(sr, self.args.rgb_range)
                    hr = utility.quantize(hr, self.args.rgb_range)
                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        eval_acc_ssim += utility.calc_ssim(
                            sr, hr, scale,
                            benchmark=self.loader_test.dataset.benchmark
                        )

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}] {:.4f}s\tPSNR: {:.3f} SSIM: {:.4f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        timer_test.release()/len(self.loader_test),
                        eval_acc / len(self.loader_test),
                        eval_acc_ssim / len(self.loader_test),
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    ))

        # self.ckp.write_log(
        #     'Total time: {:.2f}s\n'.format(timer_test.release()/len(self.loader_test)), refresh=True
        # )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))


    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]


    def crop_border(self, img_lr, img_hr, scale):
        N, C, H_lr, W_lr = img_lr.size()
        H = H_lr //2 *2
        W = W_lr //2 *2

        img_lr = img_lr[:, :, :H, :W]
        img_hr = img_hr[:, :, :round(scale * H), :round(scale * W)]

        return img_lr, img_hr


    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

